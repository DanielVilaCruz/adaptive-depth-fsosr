from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# Data
def build_index(root):
    index = {}
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        files = [f for f in sorted(d.rglob("*")) if f.suffix.lower() in IMG_EXTS]
        if files:
            index[d.name] = files
    return index


def make_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def load_batch(paths, tf):
    return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])


def sample_episode(index, rng, n_way, k_shot, n_query, n_unk_cls, n_unk_query):
    classes = rng.sample(sorted(index), n_way + n_unk_cls)
    known, unknown = classes[:n_way], classes[n_way:]
    sup, sup_y, qry, qry_y = [], [], [], []
    for c, name in enumerate(known):
        imgs = rng.sample(index[name], k_shot + n_query)
        sup += imgs[:k_shot]
        sup_y += [c] * k_shot
        qry += imgs[k_shot:]
        qry_y += [c] * n_query
    for name in unknown:
        qry += rng.sample(index[name], n_unk_query)
        qry_y += [-1] * n_unk_query
    return sup, np.array(sup_y), qry, np.array(qry_y)


# Model
def make_model(name, device):
    model = torch.hub.load("facebookresearch/dinov2", name, pretrained=True)
    return model.eval().to(device)



# Metrics
def oscr(score, pred, q_y):
    """Area under (false positive rate, correct classification rate). score: higher = more unknown."""
    known = q_y >= 0
    if known.sum() == 0 or (~known).sum() == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    correct = ((pred == q_y) & known)[order]
    unk = (~known)[order]
    ccr = np.r_[0.0, np.cumsum(correct) / known.sum()]
    fpr = np.r_[0.0, np.cumsum(unk) / (~known).sum()]
    return float(np.sum((fpr[1:] - fpr[:-1]) * (ccr[1:] + ccr[:-1]) / 2))


def evaluate(D, q_y, score=None, dual_pred=None, gate_frac=None):
    """D [Q, C] class distances (argmin = prediction); score [Q] unknown score (default min distance)."""
    known = q_y >= 0
    pred = D.argmin(1)
    s = D.min(1) if score is None else score
    res = {"acc": float((pred[known] == q_y[known]).mean()),
           "auroc": float(roc_auc_score(~known, s))}
    thr = np.percentile(s[known], 95)
    accepted = s <= thr
    res["fpr95"] = float(accepted[~known].mean())
    res["oscr"] = oscr(s, pred, q_y)
    if dual_pred is not None:
        res["dual_oscr"] = oscr(s, dual_pred, q_y)
        res["dual_ccr95"] = float((accepted & known & (dual_pred == q_y)).sum() / known.sum())
        if gate_frac is not None:
            res["cost_rel"] = float(gate_frac + (1 - gate_frac) * accepted.mean())
    return res


def _np64(t):
    return t.detach().float().cpu().numpy().astype(np.float64)
