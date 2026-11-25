import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common                                                   
from baselines import bl_args, bl_oslo, run_baselines            

COLS = ("acc", "auroc", "fpr95", "oscr")
U_PCT, U_MAX, BETA = 20.0, 0.5, 1.0 # Unknown bank (Sec. 2.3), fixed for every dataset


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


# Features
@torch.no_grad()
def extract(model, x, src, bs, device):
    G = 1 + model.num_register_tokens                   
    out = defaultdict(list)
    for i in range(0, len(x), bs):
        t = model.prepare_tokens_with_masks(x[i:i + bs].to(device))
        means = []
        for blk in model.blocks[:src]:
            t = blk(t)
            means.append(t[:, G:].mean(1))
        x_src = t
        for blk in model.blocks[src:]:
            t = blk(t)
            means.append(t[:, G:].mean(1))
        out["A"].append(torch.stack(means, 1))
        out["S"].append(x_src[:, G:].mean(1))
        out["P"].append(x_src[:, G:].half())
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def layer_tokens(model, x, cands, bs, device):
    G = 1 + model.num_register_tokens
    out = defaultdict(list)
    for i in range(0, len(x), bs):
        t = model.prepare_tokens_with_masks(x[i:i + bs].to(device))
        for L, blk in enumerate(model.blocks[:max(cands)], start=1):
            t = blk(t)
            if L in cands:
                out[L].append(t[:, G:].half())
    return {L: torch.cat(v) for L, v in out.items()}


# Patch matching (Eq. 1)
def patch_scores(qp, sp, sup_y, C, chunk=8):
    mu = sp.float().reshape(-1, sp.shape[-1]).mean(0)
    norm = lambda x: F.normalize(x.float() - mu, dim=-1).half()
    S, Np, D = sp.shape
    bank = norm(sp).reshape(S * Np, D)
    owner = torch.arange(S, device=bank.device).repeat_interleave(Np)
    cls_of = torch.as_tensor(sup_y, device=bank.device)[owner]

    def score(xp, exclude=None):
        out = []
        for i in range(0, len(xp), chunk):
            sim = (norm(xp[i:i + chunk]) @ bank.T).float()                
            if exclude is not None:
                ex = exclude[i:i + chunk]
                sim = sim.masked_fill((owner[None, :] == ex[:, None])[:, None, :], float("-inf"))
            out.append(torch.stack([sim[:, :, cls_of == c].max(-1).values.mean(1) for c in range(C)], 1))
        return torch.cat(out).double().cpu().numpy()

    return score(qp), score(sp, exclude=torch.arange(S, device=bank.device))


def calibrate(Dq, Ds, sup_y):
    own = np.eye(Ds.shape[1], dtype=bool)[sup_y]
    a, b = np.nanmean(Ds[own]), np.nanmean(Ds[~own])
    return (Dq - b) / (a - b + 1e-12)


# Depth selection (Sec. 2.2)
@torch.no_grad()
def roundtrip_score(sp, qp, sup_y, C, k_shot, keep=0.5):
    Dq, Ds = patch_scores(qp, sp, sup_y, C)
    if k_shot >= 2:
        Dq = calibrate(Dq, Ds, sup_y)
    Q = len(Dq)
    m = max(1, int(round(keep * Q / C)))
    taken = np.zeros(Q, bool)
    idx, lab = [], []
    for c in range(C):                                  
        mc = Dq[:, c] - np.delete(Dq, c, axis=1).max(1)
        mc[taken] = -np.inf
        top = np.argsort(-mc)[:m]
        taken[top] = True
        idx += top.tolist()
        lab += [c] * len(top)
    bank = qp[torch.as_tensor(idx, device=qp.device)]
    Db = patch_scores(sp, bank, np.array(lab), C)[0]     
    own = Db[np.arange(len(sup_y)), sup_y]
    other = np.where(np.eye(C, dtype=bool)[sup_y], -np.inf, Db).max(1)
    return float((Db.argmax(1) == sup_y).mean()), float((own - other).mean() / (Db.std() + 1e-12)), Dq


@torch.no_grad()
def roundtrip_all(model, sup_x, sup_y, q_x, cands, bs, device):
    y0 = np.asarray(sup_y)
    C, k_shot = int(y0.max()) + 1, int(np.bincount(y0).min())
    ts, tq = layer_tokens(model, sup_x, cands, bs, device), layer_tokens(model, q_x, cands, bs, device)
    return {L: roundtrip_score(ts[L], tq[L], y0, C, k_shot) for L in cands}


def choose_depth(base, model, args, tf, device, n_unk, test_index):
    index = base.build_index(Path(args.depth_root_cur))
    if not args.depth_same_classes:                     
        index = {c: v for c, v in index.items() if c not in test_index}
    n_uc = args.n_unknown_classes if args.depth_unknown_classes < 0 else args.depth_unknown_classes
    if args.n_way + n_uc > len(index):
        raise SystemExit(f"--depth_root {args.depth_root_cur} has {len(index)} usable classes, needs {args.n_way + n_uc}")
    rng = random.Random(args.seed + 1)
    ks = args.depth_shot or args.k_shot
    rt, acc = [], []
    for ep in range(args.depth_episodes):
        sup, sup_y, qry, q_y = base.sample_episode(index, rng, args.n_way, ks, args.n_query, n_uc, n_unk)
        r = roundtrip_all(model, base.load_batch(sup, tf), sup_y, base.load_batch(qry, tf), args.cands,
                          args.batch_size, device)
        rt.append([r[L][:2] for L in args.cands])
        if args.depth_by == "acc":                      
            y = np.asarray(q_y)
            k = y >= 0
            acc.append([float((np.round(r[L][2], 5).argmax(1)[k] == y[k]).mean()) for L in args.cands])
        print(f"[depth from {args.depth_root_cur}, {args.n_way}w{ks}s] episode {ep + 1}/{args.depth_episodes}",
              flush=True)
    R = np.asarray(rt).mean(0)
    score = [((float(a),) if acc else ()) + tuple(float(v) for v in rr)
             for a, rr in zip(np.mean(acc, 0) if acc else [0] * len(R), R)]
    best = args.cands[max(range(len(score)), key=lambda i: score[i])]
    print(f"depth for {args.n_way}w{args.k_shot}s chosen on {args.depth_root_cur}: L{best}  "
          + "  ".join(f"L{L}:{sc[0]:.3f}" for L, sc in zip(args.cands, score)), flush=True)
    return int(best), {"layer": int(best), "by": args.depth_by, "root": str(args.depth_root_cur),
                       "episodes": args.depth_episodes, "shot": ks,
                       "scores": {int(L): list(sc) for L, sc in zip(args.cands, score)}}


# Expert A: rejection (Eq. 2)
def knownness(zq, zs, sup_y, C, U=None):
    sums = np.stack([zs[sup_y == c].sum(0) for c in range(C)])
    D = zq @ l2(sums).T
    r = D.max(1)
    if U is not None and len(U):
        sim = zq @ zq[U].T
        sim[U, np.arange(len(U))] = -np.inf # Query with itself
        m = sim.max(1)
        m = np.where(np.isfinite(m), m, -np.inf)
        r = r - BETA * np.maximum(0.0, m - (zq @ zs.T).max(1))
    return D, r


def support_knownness(zs, sup_y, C, zq=None, U=None):
    sums = np.stack([zs[sup_y == c].sum(0) for c in range(C)])
    D = zs @ l2(sums).T
    D[np.arange(len(zs)), sup_y] = np.einsum("ij,ij->i", zs, l2(sums[sup_y] - zs))
    r = D.max(1)
    if U is not None and len(U):
        m = (zs @ zq[U].T).max(1)
        ss = zs @ zs.T
        np.fill_diagonal(ss, -np.inf)
        r = r - BETA * np.maximum(0.0, m - ss.max(1))
    return r


def other_class_knownness(zs, sup_y, C, zq=None, U=None):
    sums = np.stack([zs[sup_y == c].sum(0) for c in range(C)])
    D = zs @ l2(sums).T
    D[np.arange(len(zs)), sup_y] = -np.inf
    r = D.max(1)
    if U is not None and len(U):
        m = (zs @ zq[U].T).max(1)
        ss = zs @ zs.T
        np.fill_diagonal(ss, -np.inf)
        r = r - BETA * np.maximum(0.0, m - ss.max(1))
    return r


def unknown_score(zq, zs, sup_y, C, k_shot, rounds, pct_1shot):
    ref_fn, pct = (support_knownness, U_PCT) if k_shot >= 2 else (other_class_knownness, pct_1shot)
    _, r = knownness(zq, zs, sup_y, C)
    ref = ref_fn(zs, sup_y, C)
    for _ in range(rounds):
        U = np.where(r < np.percentile(ref, pct))[0]
        cap = int(round(U_MAX * len(r)))
        if len(U) > cap:
            U = U[np.argsort(r[U])[:cap]]
        _, r = knownness(zq, zs, sup_y, C, U)
        ref = ref_fn(zs, sup_y, C, zq, U)
    return -r


# Expert A: classification
@torch.no_grad()
def transductive_patch(qp, sp, sup_y, C, unk, rounds=2, keep=0.5, chunk=8):
    mu = sp.float().reshape(-1, sp.shape[-1]).mean(0)
    norm = lambda x: F.normalize(x.float() - mu, dim=-1).half()
    S, Np, D = sp.shape
    Q = len(qp)
    y = torch.as_tensor(np.asarray(sup_y), device=sp.device)
    zs, zq = norm(sp), norm(qp)
    cand = np.where(unk <= np.median(unk))[0]
    m = max(1, int(round(keep * len(cand) / C)))

    def score(bank_imgs, bank_y, bank_owner):
        bank = bank_imgs.reshape(-1, D)
        owner = bank_owner.repeat_interleave(Np)
        cls_of = bank_y.repeat_interleave(Np)
        out = []
        for i in range(0, Q, chunk):
            sim = (zq[i:i + chunk] @ bank.T).float()
            me = torch.arange(i, min(i + chunk, Q), device=sim.device) + S
            sim = sim.masked_fill((owner[None, :] == me[:, None])[:, None, :], float("-inf"))
            out.append(torch.stack([sim[:, :, cls_of == c].max(-1).values.mean(1) for c in range(C)], 1))
        return torch.cat(out).double().cpu().numpy()

    sup_owner = torch.arange(S, device=sp.device)
    Dq = score(zs, y, sup_owner)
    for _ in range(rounds):
        taken = np.zeros(Q, bool)
        idx, lab = [], []
        for c in range(C):
            mc = np.full(Q, -np.inf)
            mc[cand] = Dq[cand, c] - np.delete(Dq[cand], c, axis=1).max(1)
            mc[taken] = -np.inf
            top = [i for i in np.argsort(-mc)[:m] if np.isfinite(mc[i])]
            taken[top] = True
            idx += top
            lab += [c] * len(top)
        it = torch.as_tensor(idx, device=sp.device, dtype=torch.long)
        Dq = score(torch.cat([zs, zq[it]]), torch.cat([y, torch.as_tensor(lab, device=sp.device, dtype=y.dtype)]),
                   torch.cat([sup_owner, it + S]))
    return Dq


def expert_a(s, q, sup_y, C, args):
    mu = s["S"].float().mean(0, keepdim=True)
    nr = lambda x: F.normalize(x.float() - mu, dim=-1).double().cpu().numpy()
    u = unknown_score(nr(q["S"]), nr(s["S"]), sup_y, C, args.k_shot, args.rounds, args.u_pct_1shot)
    return transductive_patch(q["P"], s["P"], sup_y, C, u, args.t_rounds, args.t_keep), u


# Fusion
def zrow(a):
    a = np.asarray(a, float)
    return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-12)


def rank(v):
    return np.argsort(np.argsort(v)) / max(1, len(v) - 1)


def evaluate_pred(base, pred, score, q_y, C):
    return base.evaluate(-np.eye(C)[pred], q_y, score=score)     


# Episode
def run_episode(base, model, args, sup_x, sup_y, q_x, q_y, C, device):
    depth = len(model.blocks)
    sup_y = np.asarray(sup_y)
    rt = None
    if args.src == "auto":
        rt = roundtrip_all(model, sup_x, sup_y, q_x, args.cands, args.batch_size, device)
        src = max(rt, key=lambda L: rt[L][:2])
    else:
        src = int(args.src)
    s = extract(model, sup_x, src, args.batch_size, device)
    q = extract(model, q_x, src, args.batch_size, device)
    res = {}
    if not args.skip_ours:
        a, u_a = expert_a(s, q, sup_y, C, args)
        res["Ours"] = evaluate_pred(base, a.argmax(1), u_a, q_y, C)
        g = lambda L: (base._np64(q["A"][:, L - 1]), base._np64(s["A"][:, L - 1]))
        (zq12, zs12), (zqs, zss) = g(depth), g(src)
        D12, u12 = bl_oslo(zq12, zs12, sup_y, C)
        res["Ours + OSLO@L12 (fusion)"] = base.evaluate(-(zrow(a) + zrow(-D12)), q_y, score=(rank(u_a) + rank(u12)) / 2)
        cat = lambda x, y: np.concatenate([l2(x), l2(y)], 1)
        b, u_b = bl_oslo(cat(zqs, zq12), cat(zss, zs12), sup_y, C)          # Expert B
        res["OSLO on L*+L12"] = base.evaluate(b, q_y, score=u_b)
        res["Ours + OSLO on L*+L12 (fusion)"] = base.evaluate(-(zrow(a) + zrow(-b)), q_y,
                                                              score=(rank(u_a) + rank(u_b)) / 2)
    layers = {f"L{L}": L for L in args.bl_layer_list if L <= depth}
    if args.lsel_baselines:
        layers = {"Lsel": src, **layers}
    for tag, L in layers.items():
        bl_res = {}
        run_baselines(bl_args(args), {("pmean", L): (q["A"][:, L - 1], s["A"][:, L - 1])},
                           sup_y, q_y, C, None, depth, bl_res, {})
        res.update({k.replace(f"pmean@L{L}", tag): v for k, v in bl_res.items()})
        if "oslo" in args.baselines.split(","):
            D, sc = bl_oslo(base._np64(q["A"][:, L - 1]), base._np64(s["A"][:, L - 1]), sup_y, C)
            res[f"{tag}:oslo[T]"] = base.evaluate(D, q_y, score=sc)
    return res, src, ({int(L): [float(v) for v in rt[L][:2]] for L in rt} if rt else None)


# Runs
def run_setting(base, model, args, index, tf, device, n_unk):
    depth_info, src_arg = None, args.src
    if args.src == "dataset":
        L, depth_info = choose_depth(base, model, args, tf, device, n_unk, index)
        args.src = str(L)
    try:
        rng = random.Random(args.seed)                   
        agg = defaultdict(lambda: defaultdict(list))
        chosen, rts = [], []
        for ep in range(args.episodes):
            sup, sup_y, qry, q_y = base.sample_episode(index, rng, args.n_way, args.k_shot, args.n_query,
                                                       args.n_unknown_classes, n_unk)
            res, src, rt = run_episode(base, model, args, base.load_batch(sup, tf), sup_y, base.load_batch(qry, tf),
                                       q_y, args.n_way, device)
            chosen.append(int(src))
            rts.append(rt)
            for m, r in res.items():
                for k, v in r.items():
                    if k in COLS:
                        agg[m][k].append(float(v) if v is not None else float("nan"))
            print(f"[{args.n_way}w{args.k_shot}s, {n_unk} unknown queries per class] episode {ep + 1}/{args.episodes}",
                  flush=True)
    finally:
        args.src = src_arg
    out = Path(args.out_dir) / f"{args.n_way}w{args.k_shot}s" / f"u{n_unk}"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"args": {**vars(args), "n_unknown_query": n_unk}, "src_chosen": chosen,
               "roundtrip_scores": rts, "depth_from_train": depth_info,
               "methods": {m: {c: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)), "per_episode": v}
                               for c, v in r.items()} for m, r in agg.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [f"# {args.n_way}-way {args.k_shot}-shot, {n_unk} unknown queries per class, {args.episodes} episodes",
             "", f"Layer used (episodes): " + ", ".join(f"L{L}: {chosen.count(L)}" for L in sorted(set(chosen))), "",
             "| Method | " + " | ".join(COLS) + " |", "|---|" + "---|" * len(COLS)]
    lines += [f"| {m} | " + " | ".join(f"{np.nanmean(r[c]):.3f} ± {np.nanstd(r[c]):.3f}" if r.get(c) else "-"
                                       for c in COLS) + " |" for m, r in agg.items()]
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines) + f"\n\nSaved {out / 'summary.json'} and {out / 'table.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data_root", required=True, nargs="+",
                    help="dataset folders (a test/ subfolder is used if present); with several, results go to "
                         "<out_dir>/<folder name>/")
    ap.add_argument("--model", default="dinov2_vitb14_reg")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--src", default="auto", help="'auto' (depth per episode), 'dataset' (one depth per dataset, "
                                                  "from --depth_root) or a layer number")
    ap.add_argument("--src_candidates", default="4-12", help="candidate layers for the depth selection")
    ap.add_argument("--depth_root", nargs="+", default=[], help="--src dataset: one folder per --data_root, same order")
    ap.add_argument("--depth_same_classes", action="store_true",
                    help="--src dataset: --depth_root holds other images of the test classes (make_holdout.py)")
    ap.add_argument("--depth_by", default="rt", choices=["rt", "acc"],
                    help="--src dataset: rt = round trip R(L) (no labels); acc = patch-matching accuracy (their labels)")
    ap.add_argument("--depth_shot", type=int, default=5, help="--src dataset: shots of the depth episodes (0 = K)")
    ap.add_argument("--depth_episodes", type=int, default=50)
    ap.add_argument("--depth_unknown_classes", type=int, default=-1, help="-1 = --n_unknown_classes")
    ap.add_argument("--rounds", type=int, default=3, help="Expert A: unknown-bank rounds")
    ap.add_argument("--t_rounds", type=int, default=2, help="Expert A: transductive pseudo-labelling rounds")
    ap.add_argument("--t_keep", type=float, default=0.5, help="Expert A: share of likely-known queries added")
    ap.add_argument("--u_pct_1shot", type=float, default=50.0, help="1-shot unknown-bank threshold (percentile)")
    ap.add_argument("--skip_ours", action="store_true", help="only the depth selection and the baselines")
    ap.add_argument("--baselines", default="maxlogit,knn,ostim,oslo")
    ap.add_argument("--bl_layers", default="12", help="layers the baselines run at, e.g. 12 or 4-12 or 8,12")
    ap.add_argument("--lsel_baselines", action="store_true", help="also run the baselines at L* (rows Lsel:*)")
    ap.add_argument("--configs", default="5w1s,5w5s", help="comma list of <N>w<K>s")
    ap.add_argument("--n_query", type=int, default=10, help="queries per known class")
    ap.add_argument("--n_unknown_classes", type=int, default=5)
    ap.add_argument("--n_unknown_query", default="5", help="queries per unknown class; comma list allowed")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    cfgs = []
    for c in args.configs.split(","):
        n, k = c.strip().lower().rstrip("s").split("w")
        if (int(n), int(k)) not in cfgs:
            cfgs.append((int(n), int(k)))
    unk_list = [int(u) for u in str(args.n_unknown_query).split(",") if u]
    args.bl_layer_list = []
    for part in [p for p in args.bl_layers.split(",") if p]:
        a, _, b = part.partition("-")
        args.bl_layer_list += list(range(int(a), int(b or a) + 1))
    lo, _, hi = args.src_candidates.partition("-")
    args.cands = list(range(int(lo), int(hi or lo) + 1))
    if args.src == "dataset" and len(args.depth_root) != len(args.data_root):
        raise SystemExit("--src dataset needs one --depth_root per --data_root, in the same order")

    base = common
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = base.make_model(args.model, device)
    tf = base.make_transform(args.image_size)
    roots, out_base = list(args.data_root), args.out_dir
    for i, r in enumerate(roots):
        args.data_root, args.depth_root_cur = r, (args.depth_root[i] if args.src == "dataset" else None)
        args.out_dir = str(Path(out_base) / Path(r).name) if len(roots) > 1 else out_base
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        root = Path(r)
        if (root / "test").is_dir() and len([d for d in root.iterdir() if d.is_dir()]) <= 3:
            root = root / "test"
        index = base.build_index(root)
        print(f"\n=== {root} -> {args.out_dir} ===")
        for n, k in cfgs:
            if n + args.n_unknown_classes > len(index):
                print(f"Skipping {n}w{k}s: needs {n + args.n_unknown_classes} classes, {root} has {len(index)}")
                continue
            args.n_way, args.k_shot = n, k
            for u in unk_list:
                run_setting(base, model, args, index, tf, device, u)


if __name__ == "__main__":
    main()
