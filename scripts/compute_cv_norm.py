"""Global per-dim ContentVec mean/std for FM target normalization.

Computed over the TRAIN split ONLY (never val/test -> no leak). Streaming, float64
accumulation. Saves processed/contentvec_norm.npz {mean[768], std[768]} which the
model loads as buffers; FM normalizes the target by (cv-mean)/std and de-normalizes
samples by x*std+mean. Aug clips are pitch-shifted copies of train -> same distribution,
so real-train clips suffice for the stats.
"""
import os, sys, json, random
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
NPZ = BASE / "processed" / "npz"
N_CLIPS = 6000   # plenty to estimate 768-dim mean/std robustly

def main():
    rows = []
    with open(BASE / "processed" / "splits" / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    random.Random(0).shuffle(rows)

    s = np.zeros(768, np.float64); ss = np.zeros(768, np.float64); n = 0
    used = 0
    for r in rows:
        p = NPZ / r["dataset"] / f"{r['seq']:06d}.npz"
        if not p.exists():
            continue
        cv = np.load(p, allow_pickle=True)["contentvec"].astype(np.float64)  # [T,768]
        s += cv.sum(0); ss += (cv * cv).sum(0); n += cv.shape[0]
        used += 1
        if used >= N_CLIPS:
            break

    mean = s / n
    var = np.maximum(ss / n - mean * mean, 1e-8)
    std = np.sqrt(var)
    out = BASE / "processed" / "contentvec_norm.npz"
    np.savez(out, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"clips={used} frames={n:,}")
    print(f"mean: min={mean.min():.3f} max={mean.max():.3f} |mean|avg={np.abs(mean).mean():.4f}")
    print(f"std : min={std.min():.3f} max={std.max():.3f} avg={std.mean():.4f}")
    print(f"saved -> {out}")

if __name__ == "__main__":
    main()
