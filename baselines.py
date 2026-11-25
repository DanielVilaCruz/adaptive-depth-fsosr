import types
import numpy as np
from common import _np64, evaluate

BASELINES_ALL = ("knn", "msp", "maxlogit", "energy", "ostim", "oslo")
TRANSDUCTIVE = {"ostim", "oslo"}         


def _l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _logsumexp(a, axis=1):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def _protos(zs, sup_y, C):
    return _l2(np.stack([zs[sup_y == c].mean(0) for c in range(C)]))


def bl_knn(zq, zs, sup_y, C, k):
    Dall = 1 - zq @ zs.T
    if k <= 0:
        k = max(1, len(zs) // C)
    k = min(k, len(zs))
    score = np.sort(Dall, axis=1)[:, k - 1]
    D = np.stack([Dall[:, sup_y == c].min(1) for c in range(C)], axis=1)
    return D, score


def proto_logits(zq, zs, sup_y, C, temp):
    return temp * (zq @ _protos(zs, sup_y, C).T)


def bl_logit_scores(zq, zs, sup_y, C, temp, kind):
    lg = proto_logits(zq, zs, sup_y, C, temp)
    if kind == "msp":
        score = 1 - np.exp(lg.max(1) - _logsumexp(lg))
    elif kind == "maxlogit":
        score = -lg.max(1)
    else:
        score = -_logsumexp(lg)
    return -lg, score


def bl_ostim_style(zq, zs, sup_y, C, temp, iters, init_frac, sup_weight=3.0, center=True):
    if center:
        mu = np.concatenate([zq, zs]).mean(0)
        zq, zs = _l2(zq - mu), _l2(zs - mu)
    W = _protos(zs, sup_y, C)
    Q = len(zq)
    m = max(1, int(round(init_frac * Q)))
    hard = np.argsort((zq @ W.T).max(1))[:m]
    w0 = _l2(zq[hard].mean(0, keepdims=True))[0]
    log_pi = np.full(C + 1, -np.log(C + 1))
    for _ in range(iters):
        lg = temp * np.concatenate([(zq @ w0)[:, None], zq @ W.T], axis=1) + log_pi
        r = np.exp(lg - _logsumexp(lg)[:, None])
        pi = np.clip(r.mean(0), 1e-3, None)
        log_pi = np.log(pi / pi.sum())
        w0 = _l2((r[:, :1] * zq).sum(0, keepdims=True))[0]
        W = _l2(np.stack([sup_weight * zs[sup_y == c].sum(0) + (r[:, c + 1:c + 2] * zq).sum(0) for c in range(C)]))
    lg = temp * np.concatenate([(zq @ w0)[:, None], zq @ W.T], axis=1) + log_pi
    r = np.exp(lg - _logsumexp(lg)[:, None])
    return -np.log(r[:, 1:] + 1e-12), r[:, 0]


def run_baselines(args, desc, sup_y, q_y, C, dual_pred, depth, results, nbytes, qtag=""):
    names = BASELINES_ALL if args.baselines == "all" else tuple(
        b for b in args.baselines.split(",") if b)
    assert all(b in BASELINES_ALL for b in names), f"--baselines must be from {BASELINES_ALL} or 'all'"
    bl_kinds = set(args.bl_kinds.split(","))
    for (kind, L), (q, s) in desc.items():
        if kind not in bl_kinds:
            continue
        if q.shape[1] > 4096:
            continue        
        zq, zs = raw = _np64(q), _np64(s)
        if args.bl_center:                      
            mu = zs.mean(0, keepdims=True)
            zq, zs = zq - mu, zs - mu
        if not args.bl_no_l2:
            zq, zs = _l2(zq), _l2(zs)
        for b in names:
            if b == "knn":
                D, sc = bl_knn(zq, zs, sup_y, C, args.knn_k)
            elif b in ("msp", "maxlogit", "energy"):
                D, sc = bl_logit_scores(zq, zs, sup_y, C, args.bl_temp, b)
            elif b == "oslo":               
                D, sc = bl_oslo(raw[0], raw[1], sup_y, C)
            else:
                D, sc = bl_ostim_style(zq, zs, sup_y, C, args.bl_temp, args.ostim_iters, args.ostim_init_frac,
                                       args.ostim_sup_weight, not args.ostim_no_center)
            nm = f"{kind}@L{L}" + (f":{qtag}" if qtag else "") + f":{b}" + ("[T]" if b in TRANSDUCTIVE else "")
            results[nm] = evaluate(D, q_y, score=sc, dual_pred=dual_pred, gate_frac=L / depth)
            nbytes[nm] = q.shape[1] * 4


def bl_oslo(zq, zs, sup_y, C, steps=2, lambda_s=0.05, lambda_z=0.1):
    mu = np.concatenate([zs, zq]).mean(0, keepdims=True)
    zs, zq = _l2(zs - mu), _l2(zq - mu)
    Y = np.eye(C)[sup_y]
    P = np.stack([zs[sup_y == c].mean(0) for c in range(C)])
    z = np.full((len(zq), C), 1.0 / C)
    a = np.zeros((len(zq), 1))
    for _ in range(steps):
        lq = zq @ _l2(P).T
        a = (z * lq / lambda_s).sum(1, keepdims=True)
        s = 1 / (1 + np.exp(-a))
        e = s * lq / lambda_z
        z = np.exp(e - e.max(1, keepdims=True))
        z /= z.sum(1, keepdims=True)
        W = np.concatenate([Y, s * z])
        P = W.T @ np.concatenate([zs, zq]) / W.sum(0)[:, None]
    return -(zq @ _l2(P).T), -a[:, 0]


def bl_args(args):
    return types.SimpleNamespace(baselines=args.baselines,
                                 bl_kinds="pmean", bl_center=True, bl_no_l2=False, knn_k=0, bl_temp=20.0,
                                 ostim_iters=10, ostim_sup_weight=3.0, ostim_no_center=False, ostim_init_frac=0.3)
