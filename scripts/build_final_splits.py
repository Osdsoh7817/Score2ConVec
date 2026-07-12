"""Build train_final.jsonl / val_final.jsonl for the 7-language FINAL model.

LINEUP (S57 + S58 data decisions):
  zh = m4singer + gtsinger_zh            (real zh; ace_train EXCLUDED — mostly-unsampled under balance)
  ja = labs, re-segmented                (kept <=1000fr clips AS-IS + 1214 AP-re-seg sub-clips; long clips replaced)
  en = gtsinger_en                       (cons3 de-crammed)
  de/fr/es/it = mfa_*                    (cons3 de-crammed)
  DROP: ace_train, gtsinger_ja, mfa_korean, mfa_russian
LEAK-SAFE: ja train pool = train.jsonl ja-labs MINUS val_labja; ja val = val_labja. Explicit (ds,seq) leak check.
"""
import json, collections
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SPL = BASE / "processed" / "splits"
JALAB = {"kiritan", "itako", "natsume", "ofuton", "oniku", "pjs"}
# zh = m4singer ONLY (manual/native). gtsinger_zh DROPPED (S58, user): it's gtsinger-family —
# auto-aligned, NOT de-crammed (not in the 8 LOOSE ds), expressive material = the塌-prone
# gtsinger_ja risk profile; zh commits on m4singer alone (the proven zhreal_decram3 recipe). ace also out.
KEEP_NONJA = {"m4singer", "gtsinger_en",
              "mfa_german", "mfa_french", "mfa_spanish", "mfa_italian"}
MAXF = 1000


def load(name):
    return [json.loads(l) for l in open(SPL / f"{name}.jsonl", encoding="utf-8")]


def key(r):
    return (r["dataset"], r["seq"])


def main():
    train = load("train")
    val = load("val")
    val_labja = load("val_labja")
    reseg = [json.loads(l) for l in open(BASE / "processed" / "_reseg_jalab_records.jsonl", encoding="utf-8")]
    val_labja_keys = {key(r) for r in val_labja}

    # ---------- TRAIN_FINAL ----------
    train_final = []
    # non-ja keep datasets, non-aug
    for r in train:
        if r.get("aug_shift", 0) != 0:
            continue
        if r["dataset"] in KEEP_NONJA:
            train_final.append(r)
    # ja-labs: kept <=MAXF clips from pool (train.jsonl ja - val_labja); long ones replaced by sub-clips
    for r in train:
        if r.get("aug_shift", 0) != 0 or r["dataset"] not in JALAB:
            continue
        if key(r) in val_labja_keys:
            continue                      # leak-safe: never train a held-out clip
        if r.get("frames", 0) <= MAXF:
            train_final.append(r)         # kept short clip (byte-identical to labja)
        # long (>MAXF) clips are intentionally dropped here — replaced by reseg sub-clips below
    train_final.extend(reseg)             # the 1214 AP-re-segmented sub-clips

    # ---------- VAL_FINAL ----------
    val_final = []
    for r in val:
        if r.get("aug_shift", 0) != 0:
            continue
        if r["dataset"] in KEEP_NONJA:
            val_final.append(r)
    val_final.extend(val_labja)           # ja held-out

    # ---------- LEAK CHECK ----------
    tr_keys = collections.Counter(key(r) for r in train_final)
    va_keys = collections.Counter(key(r) for r in val_final)
    dup_tr = [k for k, c in tr_keys.items() if c > 1]
    dup_va = [k for k, c in va_keys.items() if c > 1]
    leak = set(tr_keys) & set(va_keys)
    print(f"train_final dup keys: {len(dup_tr)} | val_final dup keys: {len(dup_va)}")
    print(f"LEAK train_final ∩ val_final: {len(leak)} clips  {'<-- LEAK!!' if leak else '(clean)'}")
    if leak:
        for k in list(leak)[:10]:
            print("   ", k)

    # ---------- REPORT ----------
    def report(recs, name):
        by_lang = collections.defaultdict(lambda: [0, 0])  # n, frames
        by_ds = collections.Counter()
        for r in recs:
            by_lang[r["lang"]][0] += 1
            by_lang[r["lang"]][1] += r["frames"]
            by_ds[r["dataset"]] += 1
        print(f"\n=== {name}: {len(recs)} clips, {sum(v[1] for v in by_lang.values())/50/3600:.2f}h ===")
        for lang in sorted(by_lang, key=lambda l: -by_lang[l][1]):
            n, fr = by_lang[lang]
            print(f"  {lang:4s} n={n:6d}  {fr/50/3600:7.2f}h")
        print("  datasets:", dict(by_ds))

    report(train_final, "train_final")
    report(val_final, "val_final")

    if leak or dup_tr or dup_va:
        print("\n!!! NOT WRITING (leak/dup found) — fix first")
        return
    with open(SPL / "train_final.jsonl", "w", encoding="utf-8") as f:
        for r in train_final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(SPL / "val_final.jsonl", "w", encoding="utf-8") as f:
        for r in val_final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWROTE train_final.jsonl ({len(train_final)}) + val_final.jsonl ({len(val_final)})")


if __name__ == "__main__":
    main()
