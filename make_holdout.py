import argparse
import os
import random
from pathlib import Path

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dataset folder (flat <class>/ folders, or with a test/ subfolder)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.2, help="share of each class's images held out")
    ap.add_argument("--classes", type=int, default=0, help="hold out this many whole classes instead of a share of "
                                                           "each class's images (class-disjoint split)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src = Path(args.src).resolve()
    if (src / "test").is_dir() and len([d for d in src.iterdir() if d.is_dir()]) <= 3:
        src = src / "test"
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; choose another --out or remove it")
    rng = random.Random(args.seed)
    classes = sorted(p for p in src.iterdir() if p.is_dir())
    held = set()
    if args.classes:
        if len(classes) - args.classes < 10:
            raise SystemExit(f"{len(classes)} classes: holding out {args.classes} leaves fewer than 10 for test")
        held = set(rng.sample([c.name for c in classes], args.classes))
    n_tr = n_te = 0
    for c in classes:
        files = sorted(f for f in c.rglob("*") if f.suffix.lower() in EXT)
        if not files:
            continue
        rng.shuffle(files)
        if args.classes:
            k = len(files) if c.name in held else 0
        else:
            k = max(1, int(round(args.frac * len(files))))
        for part, fs in (("train", files[:k]), ("test", files[k:])):
            if not fs:
                continue
            d = out / part / c.name
            d.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(fs):
                os.symlink(f.resolve(), d / f"{i:05d}_{f.name}")
        n_tr += k
        n_te += len(files) - k
    if args.classes:
        print(f"held-out classes ({len(held)}): {', '.join(sorted(held))}; test classes: {len(classes) - len(held)}")
    print(f"{out}/train: {n_tr} held-out images (for --depth_root)\n{out}/test: {n_te} images (for --data_root {out})")


if __name__ == "__main__":
    main()
