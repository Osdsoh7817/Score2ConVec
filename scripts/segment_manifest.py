"""Segment long utterances in the manifest at SP boundaries.

Reads:  processed/manifest.jsonl
Writes: processed/manifest_seg.jsonl

Rules:
  - Utterances <= TARGET_MAX frames: kept as-is
  - Utterances > TARGET_MAX frames: split at SP phone boundaries
    - Cut AFTER the SP phone (SP stays with left segment, intact)
    - Prefer the latest SP cut that keeps segment <= TARGET_MAX
    - If no SP within budget, allow oversized segment until next SP
    - Merge trailing scraps (< MIN_SEGMENT frames) into previous segment
"""

import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_IN = os.path.join(BASE, "processed", "manifest.jsonl")
MANIFEST_OUT = os.path.join(BASE, "processed", "manifest_seg.jsonl")

TARGET_MAX = 450   # preferred max frames per segment
MIN_SEGMENT = 30   # merge trailing scraps below this


def find_sp_cuts(phones, phone_durs):
    """Return sorted list of cut-point indices (phone index AFTER each SP with dur>=2)."""
    cuts = []
    for i, (ph, dur) in enumerate(zip(phones, phone_durs)):
        if ph == "SP" and dur >= 2:
            cuts.append(i + 1)
    return cuts


def segment(phones, phone_durs, target_max=TARGET_MAX, min_segment=MIN_SEGMENT):
    """Return list of (phone_start, phone_end) tuples covering all phones."""
    n = len(phones)
    total = sum(phone_durs)

    if total <= target_max:
        return [(0, n)]

    sp_cuts = find_sp_cuts(phones, phone_durs)
    if not sp_cuts:
        return [(0, n)]

    # Cumulative frame positions: cum[i] = start frame of phone i
    cum = [0] * (n + 1)
    for i in range(n):
        cum[i + 1] = cum[i] + phone_durs[i]

    segments = []
    start = 0
    ci = 0  # pointer into sp_cuts (only advances)

    while start < n:
        remaining = cum[n] - cum[start]
        if remaining <= target_max:
            segments.append((start, n))
            break

        # Find the latest SP cut within budget from start
        best = None
        while ci < len(sp_cuts):
            cp = sp_cuts[ci]
            if cum[cp] - cum[start] <= target_max:
                best = cp
                ci += 1
            else:
                break

        if best is not None:
            segments.append((start, best))
            start = best
        else:
            # No SP within budget — take the next SP (oversized segment)
            if ci < len(sp_cuts):
                segments.append((start, sp_cuts[ci]))
                start = sp_cuts[ci]
                ci += 1
            else:
                segments.append((start, n))
                break

    # Post: merge tiny last segment into previous
    if len(segments) > 1:
        ls, le = segments[-1]
        if cum[le] - cum[ls] < min_segment:
            ps, _ = segments[-2]
            segments = segments[:-2] + [(ps, le)]

    return segments


def build_segment_record(rec, seg_idx, phone_start, phone_end, cum_frames):
    """Create a new manifest record for one segment."""
    frame_start = cum_frames[phone_start]
    frame_end = cum_frames[phone_end]
    seg_phones = rec["phones"][phone_start:phone_end]
    seg_phone_ids = rec["phone_ids"][phone_start:phone_end]
    seg_phone_durs = rec["phone_durs"][phone_start:phone_end]
    total_frames = frame_end - frame_start

    is_only = (phone_start == 0 and phone_end == len(rec["phones"]))
    seg_id = rec["utt_id"] if is_only else f"{rec['utt_id']}__s{seg_idx}"

    return {
        "dataset": rec["dataset"],
        "utt_id": rec["utt_id"],
        "seg_id": seg_id,
        "speaker": rec["speaker"],
        "language": rec["language"],
        "speaker_id": rec["speaker_id"],
        "language_id": rec["language_id"],
        "has_midi": rec["has_midi"],
        "wav_path": rec["wav_path"],
        "cv_path": rec["cv_path"],
        "f0_path": rec["f0_path"],
        "phones": seg_phones,
        "phone_ids": seg_phone_ids,
        "phone_durs": seg_phone_durs,
        "total_frames": total_frames,
        "n_phones": len(seg_phones),
        "audio_dur": round(total_frames / 50, 6),
        "frame_start": frame_start,
        "frame_end": frame_end,
    }


def main():
    with open(MANIFEST_IN, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    print(f"Input: {len(records)} records from {MANIFEST_IN}")

    out_records = []
    stats = {
        "kept_as_is": 0,
        "segmented_utts": 0,
        "segments_created": 0,
        "no_sp_skipped": 0,
        "oversized_segments": 0,
        "merged_scraps": 0,
    }
    oversized_examples = []

    for rec in records:
        phones = rec["phones"]
        phone_durs = rec["phone_durs"]
        n = len(phones)
        total = sum(phone_durs)

        # Cumulative frames for slicing
        cum = [0] * (n + 1)
        for i in range(n):
            cum[i + 1] = cum[i] + phone_durs[i]

        segs = segment(phones, phone_durs)

        if len(segs) == 1 and segs[0] == (0, n):
            if total > TARGET_MAX:
                stats["no_sp_skipped"] += 1
            else:
                stats["kept_as_is"] += 1
            seg_rec = build_segment_record(rec, 0, 0, n, cum)
            out_records.append(seg_rec)
        else:
            stats["segmented_utts"] += 1
            for si, (ps, pe) in enumerate(segs):
                seg_frames = cum[pe] - cum[ps]
                if seg_frames > TARGET_MAX:
                    stats["oversized_segments"] += 1
                    if len(oversized_examples) < 10:
                        oversized_examples.append(
                            f"  {rec['dataset']}/{rec['utt_id']} seg{si}: {seg_frames} frames"
                        )
                seg_rec = build_segment_record(rec, si, ps, pe, cum)
                out_records.append(seg_rec)
            stats["segments_created"] += len(segs)

    # Assign sequential idx
    for i, r in enumerate(out_records):
        r["idx"] = i

    # Write output
    with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print report
    print(f"\nOutput: {len(out_records)} records → {MANIFEST_OUT}")
    print(f"  ({os.path.getsize(MANIFEST_OUT) / 1024 / 1024:.1f} MB)")
    print(f"\n=== SEGMENTATION STATS ===")
    print(f"  Kept as-is (≤{TARGET_MAX} frames):   {stats['kept_as_is']:>7d}")
    print(f"  Segmented utterances:         {stats['segmented_utts']:>7d}")
    print(f"  → Total segments created:     {stats['segments_created']:>7d}")
    print(f"  No SP, kept oversized:        {stats['no_sp_skipped']:>7d}")
    print(f"  Oversized segments (>target):  {stats['oversized_segments']:>7d}")

    if oversized_examples:
        print(f"\n  Oversized examples:")
        for ex in oversized_examples:
            print(ex)

    # Frame distribution of output
    from collections import Counter
    bins = [(0, 10), (10, 50), (50, 100), (100, 200), (200, 450), (450, 500), (500, 1000), (1000, 99999)]
    labels = ["<10", "10-49", "50-99", "100-199", "200-449", "450-499", "500-999", "1000+"]
    bin_counts = Counter()
    for r in out_records:
        tf = r["total_frames"]
        for (lo, hi), label in zip(bins, labels):
            if lo <= tf < hi:
                bin_counts[label] += 1
                break

    print(f"\n=== OUTPUT FRAME DISTRIBUTION ===")
    total_out = len(out_records)
    for label in labels:
        n = bin_counts.get(label, 0)
        pct = n / total_out * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:>10s}: {n:>7d} ({pct:5.1f}%)  {bar}")

    within_500 = sum(bin_counts.get(l, 0) for l in labels if l not in ("500-999", "1000+"))
    over_500 = sum(bin_counts.get(l, 0) for l in ("500-999", "1000+"))
    print(f"\n  Within 500 frames: {within_500:>7d} ({within_500/total_out*100:.1f}%)")
    print(f"  Over 500 frames:   {over_500:>7d} ({over_500/total_out*100:.1f}%)")

    # Per-language comparison: before vs after
    from collections import defaultdict
    lang_before = Counter()
    lang_after_within = Counter()
    lang_after_total = Counter()
    for rec in records:
        lang_before[rec["language"]] += 1
    for r in out_records:
        lang_after_total[r["language"]] += 1
        if r["total_frames"] <= 500:
            lang_after_within[r["language"]] += 1

    print(f"\n=== PER-LANGUAGE: USABLE SAMPLES (≤500 frames) ===")
    print(f"  {'Lang':>4s}  {'Before seg':>10s}  {'After seg':>10s}  {'Gain':>8s}")
    for lang in sorted(lang_before.keys()):
        before_usable = sum(1 for r in records
                           if r["language"] == lang and sum(r["phone_durs"]) <= 500)
        after_usable = lang_after_within.get(lang, 0)
        gain = after_usable - before_usable
        pct = (gain / before_usable * 100) if before_usable > 0 else float("inf")
        print(f"  {lang:>4s}  {before_usable:>10d}  {after_usable:>10d}  +{gain:>6d} ({pct:+.0f}%)")


if __name__ == "__main__":
    main()
