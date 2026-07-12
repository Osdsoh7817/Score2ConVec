"""Pack all features into per-sample .npz files for S2CVDataset.

Reads:  processed/manifest_final.jsonl
Reads:  processed/contentvec/{dataset}/{utt_id}.npy  (float16, [T_full, 768])
Reads:  processed/f0/{dataset}/{utt_id}.npy           (float32, [T_full])

Writes: processed/npz/{dataset}/{seq_id}.npz

Each .npz contains:
  phonemes      [N] int64   — IPA phoneme IDs
  note_pitch    [N] int64   — MIDI note numbers (0=rest)
  phone_dur     [N] int64   — frames per phone
  note_dur      [N] int64   — frames per parent note
  note_to_phone [N] int64   — which note each phone belongs to
  contentvec    [T, 768] float16
  f0            [T] float32
  lang          str          — language code
"""

import json
import os
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(BASE, "processed", "manifest_final.jsonl")
OUT_DIR = os.path.join(BASE, "processed", "npz")


def main():
    with open(MANIFEST, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"Input: {len(records)} records")

    os.makedirs(OUT_DIR, exist_ok=True)

    # Track per-dataset sequential IDs
    ds_seq = {}
    total_written = 0
    total_bytes = 0
    errors = []
    skipped_frames = 0
    t0 = time.time()

    # Cache for contentvec/f0 arrays (avoid re-reading for segments of same utterance)
    _cv_cache_key = None
    _cv_cache_val = None
    _f0_cache_key = None
    _f0_cache_val = None

    for i, rec in enumerate(records):
        ds = rec["dataset"]
        fs, fe = rec["frame_start"], rec["frame_end"]
        total_frames = rec["total_frames"]

        # Sequential ID per dataset
        if ds not in ds_seq:
            ds_seq[ds] = 0
            os.makedirs(os.path.join(OUT_DIR, ds), exist_ok=True)
        seq = ds_seq[ds]
        ds_seq[ds] += 1

        try:
            # Load ContentVec (with caching)
            cv_key = (ds, rec["utt_id"])
            if cv_key == _cv_cache_key:
                cv_full = _cv_cache_val
            else:
                cv_full = np.load(os.path.join(BASE, rec["cv_path"]))
                _cv_cache_key = cv_key
                _cv_cache_val = cv_full

            # Load F0 (with caching)
            f0_key = (ds, rec["utt_id"])
            if f0_key == _f0_cache_key:
                f0_full = _f0_cache_val
            else:
                f0_full = np.load(os.path.join(BASE, rec["f0_path"]))
                _f0_cache_key = f0_key
                _f0_cache_val = f0_full

            # Slice to segment range
            cv = cv_full[fs:fe]
            f0 = f0_full[fs:fe]

            # Handle slight frame count mismatches (+-1 from rounding)
            if cv.shape[0] < total_frames:
                pad = total_frames - cv.shape[0]
                cv = np.pad(cv, ((0, pad), (0, 0)), mode="edge")
            elif cv.shape[0] > total_frames:
                cv = cv[:total_frames]

            if f0.shape[0] < total_frames:
                pad = total_frames - f0.shape[0]
                f0 = np.pad(f0, (0, pad), mode="edge")
            elif f0.shape[0] > total_frames:
                f0 = f0[:total_frames]

            # Pack
            out_path = os.path.join(OUT_DIR, ds, f"{seq:06d}.npz")
            np.savez(
                out_path,
                phonemes=np.array(rec["phone_ids"], dtype=np.int64),
                note_pitch=np.array(rec["note_pitch"], dtype=np.int64),
                phone_dur=np.array(rec["phone_durs"], dtype=np.int64),
                note_dur=np.array(rec["note_dur"], dtype=np.int64),
                note_to_phone=np.array(rec["note_to_phone"], dtype=np.int64),
                contentvec=cv.astype(np.float16),
                f0=f0.astype(np.float32),
                lang=rec["language"],
            )

            total_bytes += os.path.getsize(out_path)
            total_written += 1

        except Exception as e:
            if len(errors) < 20:
                errors.append(f"{rec['seg_id']}: {e}")

        if (i + 1) % 20000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            gb = total_bytes / 1e9
            print(f"  {i+1:>7d} / {len(records)}  ({rate:.0f} rec/s, {gb:.1f} GB written)")

    elapsed = time.time() - t0
    gb = total_bytes / 1e9
    print(f"\nDone in {elapsed:.1f}s ({len(records)/elapsed:.0f} rec/s)")
    print(f"Written: {total_written} .npz files, {gb:.1f} GB total")
    print(f"Output:  {OUT_DIR}")

    # Per-dataset summary
    print(f"\n{'Dataset':20s} {'Files':>7s}")
    for ds in sorted(ds_seq.keys()):
        print(f"  {ds:20s} {ds_seq[ds]:>7d}")

    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors:
            print(f"  {e}")
    else:
        print(f"\n0 errors")


if __name__ == "__main__":
    main()
