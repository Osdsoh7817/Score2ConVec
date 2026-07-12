#!/usr/bin/env python3
"""Convert all TextGrid alignments to unified frame-level IPA @50fps.

Reads 148,878 TextGrid files across 19 dataset groups.
Outputs per-dataset JSONL files to processed/alignment/.
"""

import json
import re
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from preprocessing.phoneme_vocab import (
    convert_arpabet,
    convert_ja_romaji,
    convert_mfa,
    convert_opencpop,
    convert_popcs,
    phone_to_id,
    AP_ID,
    PAD_ID,
    SP_ID,
)
from preprocessing.dict_fixes import apply_dict_fixes, split_ko_labialized

# Dataset name → language (S28: the context/lang-aware dict fixes need it)
NAME_LANG = {
    "ace_train": "zh", "ace_test": "zh", "ace_val": "zh",
    "m4singer": "zh", "popcs": "zh",
    "gtsinger_en": "en",
    "gtsinger_ja": "ja", "itako": "ja", "kiritan": "ja", "oniku": "ja",
    "ofuton": "ja", "natsume": "ja", "pjs": "ja",
    "mfa_french": "fr", "mfa_german": "de", "mfa_italian": "it",
    "mfa_korean": "ko", "mfa_russian": "ru", "mfa_spanish": "es",
}

DATASETS_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "processed" / "alignment"
FPS = 50

# ── Dataset Registry ──────────────────────────────────────────────────

REGISTRY = [
    # ACE-Opencpop (ZH)
    {"name": "ace_train", "tg_root": "ace_opencpop/wavs/train/TextGrid",
     "converter": convert_opencpop, "ap": "keep", "wav_type": "hfa"},
    {"name": "ace_test", "tg_root": "ace_opencpop/wavs/test/TextGrid",
     "converter": convert_opencpop, "ap": "keep", "wav_type": "hfa"},
    {"name": "ace_val", "tg_root": "ace_opencpop/wavs/validation/TextGrid",
     "converter": convert_opencpop, "ap": "keep", "wav_type": "hfa"},
    # M4Singer / PopCS (ZH)
    {"name": "m4singer", "tg_root": "m4singer/extracted/m4singer/TextGrid",
     "converter": convert_opencpop, "ap": "keep", "wav_type": "hfa"},
    {"name": "popcs", "tg_root": "popcs/extracted/popcs/TextGrid",
     "converter": convert_popcs, "ap": "keep", "wav_type": "hfa"},
    # GTSinger EN/JA (degrade AP→SP)
    {"name": "gtsinger_en", "tg_root": "gtsinger/English/TextGrid",
     "converter": convert_arpabet, "ap": "degrade", "wav_type": "hfa"},
    {"name": "gtsinger_ja", "tg_root": "gtsinger/Japanese/TextGrid",
     "converter": convert_ja_romaji, "ap": "degrade", "wav_type": "hfa"},
    # Japanese datasets
    {"name": "itako", "tg_root": "itako/extracted/wav/TextGrid",
     "converter": convert_ja_romaji, "ap": "keep", "wav_type": "hfa"},
    {"name": "kiritan",
     "tg_root": "kiritan/extracted/kiritan_singing/wav/TextGrid",
     "converter": convert_ja_romaji, "ap": "keep", "wav_type": "hfa"},
    {"name": "oniku",
     "tg_root": "oniku/extracted/ONIKU_KURUMI_UTAGOE_DB/TextGrid",
     "converter": convert_ja_romaji, "ap": "keep", "wav_type": "oniku_ofuton"},
    {"name": "ofuton",
     "tg_root": "ofuton/extracted/OFUTON_P_UTAGOE_DB/TextGrid",
     "converter": convert_ja_romaji, "ap": "keep", "wav_type": "oniku_ofuton"},
    {"name": "natsume",
     "tg_root": "natsume/extracted/Natsume_Singing_DB_0713/wav/TextGrid",
     "converter": convert_ja_romaji, "ap": "keep", "wav_type": "hfa"},
    {"name": "pjs", "tg_root": "pjs/extracted/PJS_corpus_ver1.1/TextGrid",
     "converter": convert_ja_romaji, "ap": "keep", "wav_type": "pjs"},
    # MFA (FR/DE/IT/KR/RU/ES, all degrade)
    {"name": "mfa_french", "tg_root": "gtsinger/mfa_aligned/French",
     "converter": convert_mfa, "ap": "degrade", "wav_type": "mfa"},
    {"name": "mfa_german", "tg_root": "gtsinger/mfa_aligned/German",
     "converter": convert_mfa, "ap": "degrade", "wav_type": "mfa"},
    {"name": "mfa_italian", "tg_root": "gtsinger/mfa_aligned/Italian",
     "converter": convert_mfa, "ap": "degrade", "wav_type": "mfa"},
    {"name": "mfa_korean", "tg_root": "gtsinger/mfa_aligned/Korean",
     "converter": convert_mfa, "ap": "degrade", "wav_type": "mfa"},
    {"name": "mfa_russian", "tg_root": "gtsinger/mfa_aligned/Russian",
     "converter": convert_mfa, "ap": "degrade", "wav_type": "mfa"},
    {"name": "mfa_spanish", "tg_root": "gtsinger/mfa_aligned/Spanish",
     "converter": convert_mfa, "ap": "degrade", "wav_type": "mfa"},
]

# ── TextGrid Parser ───────────────────────────────────────────────────

_ITEM_RE = re.compile(r"\bitem\s*\[\d+\]\s*:")
_INTERVAL_RE = re.compile(
    r"xmin\s*=\s*([\d.eE+-]+)\s*\n\s*xmax\s*=\s*([\d.eE+-]+)\s*\n"
    r'\s*text\s*=\s*"([^"]*)"'
)


def parse_textgrid(path: Path) -> list[tuple[str, float, float]]:
    """Parse TextGrid → list of (phone_label, start_sec, end_sec)."""
    content = path.read_text(encoding="utf-8")

    m = re.search(r"^\s*size\s*=\s*(\d+)", content, re.MULTILINE)
    tier_count = int(m.group(1))
    tier_name = "word - phones" if tier_count == 4 else "phones"

    tier_starts = [m.start() for m in _ITEM_RE.finditer(content)]
    tier_starts.append(len(content))

    for i in range(len(tier_starts) - 1):
        section = content[tier_starts[i] : tier_starts[i + 1]]
        if f'name = "{tier_name}"' in section:
            return [
                (m.group(3).strip(), float(m.group(1)), float(m.group(2)))
                for m in _INTERVAL_RE.finditer(section)
            ]

    raise ValueError(f"Tier '{tier_name}' not found in {path}")


# ── WAV Path Resolution ──────────────────────────────────────────────


def tg_to_wav(tg_path: Path, wav_type: str) -> Path:
    """TextGrid path → corresponding WAV path."""
    if wav_type == "hfa":
        s = str(tg_path)
        s = s.replace("\\TextGrid\\", "\\").replace("/TextGrid/", "/")
        return Path(s).with_suffix(".wav")
    if wav_type == "mfa":
        s = str(tg_path)
        s = s.replace("\\mfa_aligned\\", "\\").replace("/mfa_aligned/", "/")
        return Path(s).with_suffix(".wav")
    if wav_type == "oniku_ofuton":
        # WAV at {base}/{stem}/{stem}.wav
        base = tg_path.parent.parent  # up from TextGrid/
        stem = tg_path.stem
        return base / stem / (stem + ".wav")
    if wav_type == "pjs":
        # WAV at {base}/{pjsXXX}/{stem}.wav
        base = tg_path.parent.parent
        stem = tg_path.stem           # e.g. pjs001_song
        subdir = stem.split("_")[0]   # e.g. pjs001
        return base / subdir / (stem + ".wav")
    raise ValueError(f"Unknown wav_type: {wav_type}")


# ── Frame Quantization ────────────────────────────────────────────────


def quantize(
    intervals: list[tuple[str, float, float]],
    converter,
    ap_policy: str,
    lang: str,
) -> tuple[list[str], list[int], list[int], int]:
    """Convert intervals → (phones, phone_ids, phone_durs, total_frames)."""
    if not intervals:
        return [], [], [], 0

    audio_dur = intervals[-1][2]
    total_frames = max(1, round(audio_dur * FPS))

    phones: list[str] = []
    phone_durs: list[int] = []

    for text, start, end in intervals:
        ipa = converter(text)
        if ap_policy == "degrade" and ipa == "AP":
            ipa = "SP"

        start_f = round(start * FPS)
        end_f = round(end * FPS)
        dur = max(1, end_f - start_f)

        phones.append(ipa)
        phone_durs.append(dur)

    # S28 dict eradication — context/lang-aware fixes (single source of truth,
    # shared with the npz patcher + inference). A2 (en ah/ax) is already handled
    # per-phone by convert_arpabet; A3 (van→yɛn) by the vocab rename.
    phones = apply_dict_fixes(phones, lang)                                # A1/C1/C2/C3/D
    phones, phone_durs, _ = split_ko_labialized(phones, phone_durs, lang)  # C4 ko Cʷ→C+w
    phone_ids = [phone_to_id(p) for p in phones]

    # Reconcile sum(durs) == total_frames
    current = sum(phone_durs)
    if current > total_frames:
        excess = current - total_frames
        idx_by_dur = sorted(range(len(phone_durs)),
                            key=lambda i: phone_durs[i], reverse=True)
        i = 0
        while excess > 0:
            j = idx_by_dur[i % len(idx_by_dur)]
            if phone_durs[j] > 1:
                phone_durs[j] -= 1
                excess -= 1
            i += 1
            if i >= len(idx_by_dur) * 50:
                break
    elif current < total_frames:
        phone_durs[-1] += total_frames - current

    return phones, phone_ids, phone_durs, total_frames


# ── Per-dataset Processing ────────────────────────────────────────────


def process_dataset(entry: dict) -> dict:
    """Process one dataset → JSONL. Returns stats."""
    name = entry["name"]
    tg_root = DATASETS_DIR / entry["tg_root"]
    converter = entry["converter"]
    ap_policy = entry["ap"]
    wav_type = entry["wav_type"]
    lang = NAME_LANG[name]

    if not tg_root.exists():
        print(f"  SKIP: {tg_root} not found")
        return {"name": name, "tg_count": 0, "written": 0, "errors": 0,
                "pad_count": 0, "ap_count": 0, "long_dur": 0, "error_details": []}

    tg_files = sorted(tg_root.rglob("*.TextGrid"))
    total = len(tg_files)
    out_path = OUTPUT_DIR / f"{name}.jsonl"

    errors: list[str] = []
    pad_count = 0
    ap_count = 0
    long_dur = 0
    dur_mismatch = 0
    written = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for i, tg_path in enumerate(tg_files):
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{total}...")

            try:
                intervals = parse_textgrid(tg_path)
                phones, pids, durs, n_frames = quantize(
                    intervals, converter, ap_policy, lang
                )

                utt_id = (
                    str(tg_path.relative_to(tg_root))
                    .replace("\\", "/")
                    .removesuffix(".TextGrid")
                )

                wav_path = tg_to_wav(tg_path, wav_type)
                wav_rel = str(wav_path.relative_to(DATASETS_DIR)).replace("\\", "/")

                # --- checks ---
                if PAD_ID in pids:
                    bad = [phones[j] for j, p in enumerate(pids) if p == PAD_ID]
                    errors.append(f"{utt_id}: unmapped {bad}")
                    pad_count += 1

                if AP_ID in pids:
                    ap_count += 1

                if durs and max(durs) > 500:
                    long_dur += 1

                if sum(durs) != n_frames:
                    dur_mismatch += 1

                if not wav_path.exists():
                    errors.append(f"{utt_id}: WAV missing {wav_rel}")

                record = {
                    "utt_id": utt_id,
                    "wav_path": wav_rel,
                    "audio_dur": round(intervals[-1][2], 6) if intervals else 0.0,
                    "phones": phones,
                    "phone_ids": pids,
                    "phone_durs": durs,
                    "total_frames": n_frames,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

            except Exception as e:
                errors.append(f"{tg_path.name}: {e}")

    return {
        "name": name,
        "tg_count": total,
        "written": written,
        "errors": len(errors),
        "pad_count": pad_count,
        "ap_count": ap_count,
        "long_dur": long_dur,
        "dur_mismatch": dur_mismatch,
        "error_details": errors[:10],
    }


# ── Main ──────────────────────────────────────────────────────────────


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}\n")

    all_stats: list[dict] = []
    t0 = time.time()

    for entry in REGISTRY:
        t1 = time.time()
        print(f"[{entry['name']}]")
        stats = process_dataset(entry)
        all_stats.append(stats)
        dt = time.time() - t1
        print(
            f"  {stats['written']}/{stats['tg_count']} "
            f"({dt:.1f}s) "
            f"err={stats['errors']} pad={stats['pad_count']} "
            f"ap={stats['ap_count']} long={stats['long_dur']} "
            f"dur_mm={stats['dur_mismatch']}"
        )
        for e in stats["error_details"]:
            print(f"  ERR: {e}")
        print()

    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 64)
    total_w = sum(s["written"] for s in all_stats)
    total_t = sum(s["tg_count"] for s in all_stats)
    total_err = sum(s["errors"] for s in all_stats)
    total_pad = sum(s["pad_count"] for s in all_stats)
    total_long = sum(s["long_dur"] for s in all_stats)
    total_mm = sum(s["dur_mismatch"] for s in all_stats)
    print(f"Total: {total_w}/{total_t} ({elapsed:.1f}s)")
    print(f"Errors: {total_err}  Unmapped(PAD): {total_pad}  "
          f"Long>10s: {total_long}  DurMismatch: {total_mm}")
    print()

    # ── Verification ──────────────────────────────────────────────────

    ok = True

    # 1. Count match
    mm = [(s["name"], s["tg_count"], s["written"])
          for s in all_stats if s["tg_count"] != s["written"]]
    if mm:
        ok = False
        print("FAIL count mismatch:")
        for n, exp, act in mm:
            print(f"  {n}: {exp} TG -> {act} written")
    else:
        print("PASS  counts match")

    # 2. No unmapped phones
    if total_pad:
        ok = False
        print(f"FAIL  {total_pad} utterances have unmapped phones (PAD)")
    else:
        print("PASS  no unmapped phones")

    # 3. Long durations
    if total_long:
        print(f"WARN  {total_long} utterances have phone dur > 10s")
    else:
        print("PASS  no long durations")

    # 4. Duration sum consistency
    if total_mm:
        ok = False
        print(f"FAIL  {total_mm} utterances have dur sum != total_frames")
    else:
        print("PASS  all dur sums match total_frames")

    # 5. GTSinger: 0 AP after degrade
    gt = [s for s in all_stats if s["name"].startswith("gtsinger")]
    gt_ap = sum(s["ap_count"] for s in gt)
    if gt_ap:
        ok = False
        print(f"FAIL  GTSinger has {gt_ap} utterances with AP (should be 0)")
    else:
        print("PASS  GTSinger: 0 AP (degrade worked)")

    # 6. MFA: 0 AP
    mfa = [s for s in all_stats if s["name"].startswith("mfa")]
    mfa_ap = sum(s["ap_count"] for s in mfa)
    if mfa_ap:
        ok = False
        print(f"FAIL  MFA has {mfa_ap} utterances with AP (should be 0)")
    else:
        print("PASS  MFA: 0 AP")

    # 7. Keep-AP datasets should have some AP
    keep = [s for s in all_stats
            if not s["name"].startswith(("gtsinger", "mfa")) and s["tg_count"] > 0]
    no_ap = [s["name"] for s in keep if s["ap_count"] == 0]
    if no_ap:
        print(f"WARN  keep-AP datasets with 0 AP: {no_ap}")
    else:
        print("PASS  all keep-AP datasets have AP tokens")

    print()
    if ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — review above")


if __name__ == "__main__":
    main()
