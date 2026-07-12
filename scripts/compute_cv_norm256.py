"""Global per-dim mean/std for the 256-d vec256l9 cv (so-vits-svc 4.0 retarget).

Mirrors compute_cv_norm.py but: reads processed/npz256 over train_final.jsonl, dim 256,
saves processed/contentvec256_norm.npz {mean[256], std[256]}. Run AFTER pack_npz256 (+reseg256).
"""
import sys, json, random
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
NPZ = BASE / "processed" / "npz256"
DIM = 256
N_CLIPS = 6000


def main():
    rows = []
    with open(BASE / "processed" / "splits" / "train_final.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    random.Random(0).shuffle(rows)

    s = np.zeros(DIM, np.float64); ss = np.zeros(DIM, np.float64); n = 0; used = 0; miss = 0
    for r in rows:
        p = NPZ / r["dataset"] / f"{r['seq']:06d}.npz"
        if not p.exists():
            miss += 1
            continue
        cv = np.load(p, allow_pickle=True)["contentvec"].astype(np.float64)  # [T,256]
        assert cv.shape[1] == DIM, f"{p}: dim {cv.shape[1]} != {DIM}"
        s += cv.sum(0); ss += (cv * cv).sum(0); n += cv.shape[0]; used += 1
        if used >= N_CLIPS:
            break

    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean * mean, 1e-8))
    out = BASE / "processed" / "contentvec256_norm.npz"
    np.savez(out, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"clips={used} frames={n:,} missing={miss}")
    print(f"mean: min={mean.min():.3f} max={mean.max():.3f} |mean|avg={np.abs(mean).mean():.4f}")
    print(f"std : min={std.min():.3f} max={std.max():.3f} avg={std.mean():.4f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
