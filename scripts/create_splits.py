"""Create train/val/test splits.

Strategy:
  - ace_train → all train
  - ace_val → all val
  - ace_test → all test
  - Small JA (itako, kiritan, oniku, ofuton, natsume, pjs) → all train
  - Everything else: stratified by speaker, ~90/5/5 per speaker group

Output: processed/splits/{train,val,test}.jsonl
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

BASE = Path(__file__).resolve().parent.parent
MANIFEST = BASE / "processed" / "manifest_final.jsonl"
OUT_DIR = BASE / "processed" / "splits"

ALL_TRAIN = {"ace_train", "itako", "kiritan", "oniku", "ofuton", "natsume", "pjs"}
ALL_VAL = {"ace_val"}
ALL_TEST = {"ace_test"}

VAL_RATIO = 0.05
TEST_RATIO = 0.05

MAX_CONTENT_PHONE_DUR = 200  # 4 seconds at 50fps — anything longer is alignment failure


def main():
    OUT_DIR.mkdir(exist_ok=True)

    with open(MANIFEST, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"Total records: {len(records)}")

    filtered = 0
    ds_seq = {}
    samples = []
    for rec in records:
        has_extreme = any(
            dur > MAX_CONTENT_PHONE_DUR
            for ph, dur in zip(rec["phones"], rec["phone_durs"])
            if ph not in ("SP", "AP")
        )
        if has_extreme:
            filtered += 1
            # Still count for seq numbering so NPZ indices stay aligned
            ds = rec["dataset"]
            if ds not in ds_seq:
                ds_seq[ds] = 0
            ds_seq[ds] += 1
            continue
        ds = rec["dataset"]
        if ds not in ds_seq:
            ds_seq[ds] = 0
        seq = ds_seq[ds]
        ds_seq[ds] += 1

        samples.append({
            "dataset": ds,
            "seq": seq,
            "frames": rec["total_frames"],
            "n_phones": rec["n_phones"],
            "lang": rec["language"],
            "speaker_id": rec["speaker_id"],
        })

    train, val, test = [], [], []

    ds_samples = defaultdict(list)
    for s in samples:
        ds_samples[s["dataset"]].append(s)

    for ds, ds_list in sorted(ds_samples.items()):
        if ds in ALL_TRAIN:
            train.extend(ds_list)
        elif ds in ALL_VAL:
            val.extend(ds_list)
        elif ds in ALL_TEST:
            test.extend(ds_list)
        else:
            spk_groups = defaultdict(list)
            for s in ds_list:
                spk_groups[s["speaker_id"]].append(s)

            for spk_id, spk_samples in sorted(spk_groups.items()):
                random.shuffle(spk_samples)
                n = len(spk_samples)
                n_val = max(1, int(n * VAL_RATIO))
                n_test = max(1, int(n * TEST_RATIO))

                val.extend(spk_samples[:n_val])
                test.extend(spk_samples[n_val:n_val + n_test])
                train.extend(spk_samples[n_val + n_test:])

    if filtered:
        print(f"Filtered {filtered} records with extreme phone durations (>{MAX_CONTENT_PHONE_DUR} frames)")

    for name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = OUT_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in split_data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        langs = defaultdict(int)
        spks = set()
        total_frames = 0
        for s in split_data:
            langs[s["lang"]] += 1
            spks.add(s["speaker_id"])
            total_frames += s["frames"]

        hours = total_frames / 50 / 3600
        print(f"\n{name}: {len(split_data):,} samples, {len(spks)} speakers, {hours:.1f}h")
        for lang in sorted(langs.keys()):
            print(f"  {lang}: {langs[lang]:,}")

    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
