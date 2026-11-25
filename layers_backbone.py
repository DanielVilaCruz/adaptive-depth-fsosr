import argparse
import importlib.util
import json
import random
import sys
import timm
import numpy as np
import torch
from torchvision import transforms
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common                                     
from baselines import bl_oslo                       
COLS = ("acc", "oscr", "auroc")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_model(name, image_size, device):
    model = timm.create_model(name, pretrained=True, img_size=image_size, num_classes=0)
    cfg = model.pretrained_cfg
    return model.eval().to(device), tuple(cfg.get("mean", (0.485, 0.456, 0.406))), \
        tuple(cfg.get("std", (0.229, 0.224, 0.225)))


def make_transform(image_size, mean, std):
    return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(),
                               transforms.Normalize(mean, std)])


@torch.no_grad()
def patch_means(model, x, bs, device, tokens=None):
    P = model.num_prefix_tokens
    out = []
    for i in range(0, len(x), bs):
        t = model.patch_embed(x[i:i + bs].to(device))
        t = model._pos_embed(t)
        t = model.patch_drop(t)
        t = model.norm_pre(t)
        per, tok = [], []
        for blk in model.blocks:
            t = blk(t)
            per.append(t[:, P:].float().mean(1))
            if tokens is not None:
                tok.append(t[:, P:].half())
        out.append(torch.stack(per, 1))
        if tokens is not None:
            tokens.append(torch.stack(tok, 1))
    return torch.cat(out).double().cpu().numpy()


def run_setting(base, fs, patch, model, tf, index, args, n, k, u, device):
    rng = random.Random(args.seed)                      
    depth = len(model.blocks)
    agg = {L: {c: [] for c in COLS} for L in range(1, depth + 1)}
    pagg = {L: [] for L in range(1, depth + 1)}         
    rts = []                                          
    for ep in range(args.episodes):
        sup, sup_y, qry, q_y = base.sample_episode(index, rng, n, k, args.n_query, args.n_unknown_classes, u)
        ts, tq = ([] if patch else None), ([] if patch else None)
        s = patch_means(model, base.load_batch(sup, tf), args.batch_size, device, ts)
        q = patch_means(model, base.load_batch(qry, tf), args.batch_size, device, tq)
        if patch:
            ts, tq = torch.cat(ts), torch.cat(tq)        
            known = q_y >= 0
            for L in range(1, depth + 1):
                Dq, Ds = fs.patch_scores(tq[:, L - 1], ts[:, L - 1], np.asarray(sup_y), n)
                if k >= 2:
                    Dq = fs.calibrate(Dq, Ds, np.asarray(sup_y))
                pagg[L].append(float((Dq.argmax(1)[known] == q_y[known]).mean()))
            if args.roundtrip:
                rts.append({str(L): list(fs.roundtrip_score(ts[:, L - 1], tq[:, L - 1], np.asarray(sup_y), n, k)[:2])
                            for L in range(1, depth + 1)})
        for L in range(1, depth + 1):
            D, sc = bl_oslo(q[:, L - 1], s[:, L - 1], np.asarray(sup_y), n)
            r = base.evaluate(D, q_y, score=sc)
            for c in COLS:
                agg[L][c].append(float(r[c]) if r.get(c) is not None else float("nan"))
        print(f"[{n}w{k}s, u{u}] episode {ep + 1}/{args.episodes}", flush=True)
    out = Path(args.out_dir) / f"{n}w{k}s" / f"u{u}"
    out.mkdir(parents=True, exist_ok=True)
    methods = {f"L{L}:oslo[T]": {c: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)), "per_episode": v}
                                 for c, v in agg[L].items()} for L in agg}
    if patch:
        methods.update({f"L{L}:patch": {"acc": {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_episode": v}}
                        for L, v in pagg.items()})
    json.dump({"args": {**vars(args), "n_way": n, "k_shot": k, "n_unknown_query": u}, "methods": methods,
               **({"roundtrip_scores": rts} if rts else {})},
              open(out / "summary.json", "w"), indent=2)
    print(f"\n{args.model} | {Path(args.data_root).name} | {n}w{k}s u{u} | OSLO per layer (mean +- 95% CI)")
    best = max(agg, key=lambda L: np.nanmean(agg[L]["acc"]))
    for L in agg:
        a, o = (100 * np.asarray(agg[L][c]) for c in ("acc", "oscr"))
        ci = lambda v: 1.96 * np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum())
        pa = 100 * np.asarray(pagg[L]) if patch else None
        print(f"  L{L:2d}  acc {np.nanmean(a):5.1f} +- {ci(a):.1f}   oscr {np.nanmean(o):5.1f} +- {ci(o):.1f}"
              + (f"   patch acc {pa.mean():5.1f} +- {ci(pa):.1f}" if patch else "")
              + ("   <- best OSLO acc" if L == best else ""))
    print(f"saved {out / 'summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_script", default=str(HERE / "fsosr.py"))
    ap.add_argument("--data_root", required=True, nargs="+")
    ap.add_argument("--model", nargs="+", default=["vit_base_patch16_224.augreg_in21k_ft_in1k",
                                                   "vit_base_patch16_clip_224.openai"],
                    help="one or more timm model names; with several, results go to <out_dir>/<short name>/")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--configs", default="5w1s,5w5s")
    ap.add_argument("--n_query", type=int, default=10)
    ap.add_argument("--n_unknown_classes", type=int, default=5)
    ap.add_argument("--n_unknown_query", default="5")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="layers_vit")
    ap.add_argument("--no_patch", action="store_true", help="skip our patch-matching rows (OSLO only; faster)")
    ap.add_argument("--roundtrip", action="store_true",
                    help="also save the round-trip score R(L) of every layer per episode (for fig_depth_backbones.py)")
    args = ap.parse_args()
    assert not (args.roundtrip and args.no_patch), "--roundtrip needs the patch tokens: drop --no_patch"

    base = common
    fs = load(args.final_script, "fsosr")               
    patch = not args.no_patch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfgs = [tuple(int(v) for v in c.strip().lower().rstrip("s").split("w")) for c in args.configs.split(",")]
    models, roots, out_root = list(args.model), list(args.data_root), args.out_dir
    for name in models:
        model, mean, std = make_model(name, args.image_size, device)
        tf = make_transform(args.image_size, mean, std)
        args.model = name
        out_base = str(Path(out_root) / short_name(name)) if len(models) > 1 else out_root
        for r in roots:
            root = Path(r)
            if (root / "test").is_dir() and len([d for d in root.iterdir() if d.is_dir()]) <= 3:
                root = root / "test"                    
            index = base.build_index(root)
            args.data_root = r
            args.out_dir = str(Path(out_base) / Path(r).name) if len(roots) > 1 else out_base
            for n, k in cfgs:
                for u in (int(v) for v in str(args.n_unknown_query).split(",")):
                    run_setting(base, fs, patch, model, tf, index, args, n, k, u, device)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


def short_name(name):
    n = name.lower()
    for key, short in (("dinov2", "dinov2"), ("clip", "clip"), (".dino", "dino"), ("augreg", "vit_sup")):
        if key in n:
            return short
    return n.replace(".", "_")


if __name__ == "__main__":
    main()
