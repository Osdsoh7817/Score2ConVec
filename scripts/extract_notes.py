"""Extract note_pitch, note_dur, note_to_phone for every record in the manifest.

Extraction paths:
  A) ACE-Opencpop     — JSON metadata with note_midi / note timing
  B) M4Singer         — binary .mid files parsed with mido
  C) GTSinger/MFA     — per-sample JSON with word->note arrays
  D) JA singing DBs   — standard .mid files (itako/kiritan/oniku/ofuton/natsume/pjs)

Excluded: popcs (no MIDI annotations, F0-inferred proved unreliable)

Reads:  processed/manifest_seg.jsonl
Writes: processed/manifest_final.jsonl  (same + note_pitch, note_dur, note_to_phone)
"""

import json
import math
import os
import sys
import time

import mido
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_IN = os.path.join(BASE, "processed", "manifest_seg.jsonl")
MANIFEST_OUT = os.path.join(BASE, "processed", "manifest_final.jsonl")

# Dataset -> language directory name for GTSinger JSON lookup
GT_LANG_DIR = {
    "gtsinger_en": "English",
    "gtsinger_ja": "Japanese",
    "mfa_french": "French",
    "mfa_german": "German",
    "mfa_italian": "Italian",
    "mfa_korean": "Korean",
    "mfa_russian": "Russian",
    "mfa_spanish": "Spanish",
}

ACE_SPLIT_DIR = {
    "ace_train": "train",
    "ace_test": "test",
    "ace_val": "validation",
}

JA_MIDI_DATASETS = {
    "itako": lambda uid: os.path.join(BASE, "datasets", "itako", "extracted", "midi_label", uid + ".mid"),
    "kiritan": lambda uid: os.path.join(BASE, "datasets", "kiritan", "extracted", "kiritan_singing", "midi_label", uid + ".mid"),
    "oniku": lambda uid: os.path.join(BASE, "datasets", "oniku", "extracted", "ONIKU_KURUMI_UTAGOE_DB", uid, uid + ".mid"),
    "ofuton": lambda uid: os.path.join(BASE, "datasets", "ofuton", "extracted", "OFUTON_P_UTAGOE_DB", uid, uid + ".mid"),
    "natsume": lambda uid: os.path.join(BASE, "datasets", "natsume", "extracted", "Natsume_Singing_DB_0713", "midi", uid + ".mid"),
    "pjs": lambda uid: os.path.join(BASE, "datasets", "pjs", "extracted", "PJS_corpus_ver1.1", uid.replace("_song", ""), uid.replace("_song", "") + ".mid"),
}

F0_DATASETS = set()  # All datasets now use real MIDI annotations

EXCLUDED_DATASETS = {"popcs"}  # No MIDI annotations, F0-inferred is unreliable


# ════════════════════════════════════════════════════════
#  Note readers — each returns [(midi, start_sec, end_sec), ...]
# ════════════════════════════════════════════════════════

def read_ace_notes(utt_id, dataset):
    split_dir = ACE_SPLIT_DIR[dataset]
    path = os.path.join(BASE, "datasets", "ace_opencpop", "metadata", split_dir, f"{utt_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return [
        (int(m), s, e)
        for m, s, e in zip(meta["note_midi"], meta["note_start_times"], meta["note_end_times"])
    ]


def read_m4singer_notes(utt_id):
    path = os.path.join(BASE, "datasets", "m4singer", "extracted", "m4singer", f"{utt_id}.mid")
    mid = mido.MidiFile(path)
    tempo = 500000  # default: 120 BPM

    events = []
    for track in mid.tracks:
        abs_sec = 0.0
        active = {}
        for msg in track:
            abs_sec += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = abs_sec
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active:
                    events.append((msg.note, active.pop(msg.note), abs_sec))

    events.sort(key=lambda x: x[1])
    return events


def read_midi_file(path):
    """Read note events from a standard MIDI file."""
    mid = mido.MidiFile(path)
    tempo = 500000
    events = []
    for track in mid.tracks:
        abs_sec = 0.0
        active = {}
        for msg in track:
            abs_sec += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = abs_sec
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active:
                    events.append((msg.note, active.pop(msg.note), abs_sec))
    events.sort(key=lambda x: x[1])
    return events


def read_gtsinger_notes(utt_id, lang_dir):
    path = os.path.join(BASE, "datasets", "gtsinger", lang_dir, f"{utt_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        words = json.load(f)
    if not words or "note" not in words[0]:
        return None  # Paired_Speech_Group — no note annotations, fall back to F0
    events = []
    for word in words:
        for midi, start, end in zip(word["note"], word["note_start"], word["note_end"]):
            events.append((int(midi), float(start), float(end)))
    return events


def infer_notes_from_f0(f0, phones, phone_durs, frame_start):
    """Per-phone MIDI from F0 contour. Returns list of int (len = len(phones))."""
    per_phone = []
    offset = 0
    for ph, dur in zip(phones, phone_durs):
        if ph in ("SP", "AP"):
            per_phone.append(0)
        else:
            f0_slice = f0[frame_start + offset: frame_start + offset + dur]
            voiced = f0_slice[f0_slice > 30]
            if len(voiced) > 0:
                med = float(np.median(voiced))
                midi = int(round(12 * math.log2(med / 440) + 69))
                per_phone.append(max(1, min(127, midi)))
            else:
                per_phone.append(0)
        offset += dur
    return per_phone


# ════════════════════════════════════════════════════════
#  Timing-based note->phone mapping
# ════════════════════════════════════════════════════════

def map_notes_to_phones(note_events, phones, phone_durs, frame_start):
    """Assign a MIDI note to each phone via maximum time overlap."""
    n = len(phones)
    cum = [0] * (n + 1)
    for i in range(n):
        cum[i + 1] = cum[i] + phone_durs[i]

    per_phone = []
    for i in range(n):
        if phones[i] in ("SP", "AP"):
            per_phone.append(0)
            continue

        ph_s = (frame_start + cum[i]) / 50.0
        ph_e = (frame_start + cum[i + 1]) / 50.0
        if ph_e <= ph_s:
            per_phone.append(0)
            continue

        best_midi, best_ov = 0, 0.0
        for midi, ns, ne in note_events:
            if ne <= ph_s or ns >= ph_e:
                continue
            ov = min(ph_e, ne) - max(ph_s, ns)
            if ov > best_ov:
                best_ov = ov
                best_midi = midi
        per_phone.append(best_midi)

    return per_phone


# ════════════════════════════════════════════════════════
#  Build note_dur / note_to_phone from per-phone MIDI
# ════════════════════════════════════════════════════════

def build_note_arrays(per_phone_midi, phone_durs):
    """Group consecutive same-MIDI phones into notes."""
    n = len(per_phone_midi)
    note_pitch = list(per_phone_midi)
    note_to_phone = [0] * n
    note_dur = [0] * n

    # Find group boundaries
    groups = []
    gs = 0
    for i in range(1, n):
        if per_phone_midi[i] != per_phone_midi[gs]:
            groups.append((gs, i))
            gs = i
    groups.append((gs, n))

    for note_idx, (gs, ge) in enumerate(groups):
        g_dur = sum(phone_durs[gs:ge])
        for j in range(gs, ge):
            note_to_phone[j] = note_idx
            note_dur[j] = g_dur

    return note_pitch, note_dur, note_to_phone


# ════════════════════════════════════════════════════════
#  Read note events with caching (avoid re-reading for segments)
# ════════════════════════════════════════════════════════

_note_cache = {}
_f0_cache = {}


def get_note_events(rec):
    """Return note events for the ORIGINAL utterance (cached)."""
    ds = rec["dataset"]
    utt_id = rec["utt_id"]
    key = (ds, utt_id)

    if key in _note_cache:
        return _note_cache[key]

    if ds in ACE_SPLIT_DIR:
        events = read_ace_notes(utt_id, ds)
    elif ds == "m4singer":
        events = read_m4singer_notes(utt_id)
    elif ds in GT_LANG_DIR:
        events = read_gtsinger_notes(utt_id, GT_LANG_DIR[ds])
    elif ds in JA_MIDI_DATASETS:
        midi_path = JA_MIDI_DATASETS[ds](utt_id)
        events = read_midi_file(midi_path) if os.path.exists(midi_path) else None
    else:
        events = None  # F0-inferred path

    _note_cache[key] = events
    if len(_note_cache) > 200:
        oldest = next(iter(_note_cache))
        del _note_cache[oldest]

    return events


def get_f0(rec):
    """Load F0 array for F0-inferred datasets (cached)."""
    ds = rec["dataset"]
    utt_id = rec["utt_id"]
    key = (ds, utt_id)

    if key in _f0_cache:
        return _f0_cache[key]

    f0_path = os.path.join(BASE, rec["f0_path"])
    f0 = np.load(f0_path)
    _f0_cache[key] = f0

    if len(_f0_cache) > 200:
        oldest = next(iter(_f0_cache))
        del _f0_cache[oldest]

    return f0


# ════════════════════════════════════════════════════════
#  Process one record
# ════════════════════════════════════════════════════════

def process_record(rec):
    """Return (note_pitch, note_dur, note_to_phone) for one manifest record."""
    phones = rec["phones"]
    phone_durs = rec["phone_durs"]
    frame_start = rec["frame_start"]
    ds = rec["dataset"]

    if ds in F0_DATASETS:
        f0 = get_f0(rec)
        per_phone = infer_notes_from_f0(f0, phones, phone_durs, frame_start)
    else:
        note_events = get_note_events(rec)
        if note_events is None:
            # GTSinger Paired_Speech_Group or other missing annotations -> F0 fallback
            f0 = get_f0(rec)
            per_phone = infer_notes_from_f0(f0, phones, phone_durs, frame_start)
        else:
            per_phone = map_notes_to_phones(note_events, phones, phone_durs, frame_start)

    return build_note_arrays(per_phone, phone_durs)


# ════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════

def main():
    with open(MANIFEST_IN, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"Input: {len(records)} records")

    from collections import Counter
    ds_counts = Counter()
    ds_errors = Counter()
    issues = []
    t0 = time.time()

    out_records = []
    skipped = 0
    for i, rec in enumerate(records):
        if rec["dataset"] in EXCLUDED_DATASETS:
            skipped += 1
            continue
        try:
            note_pitch, note_dur, note_to_phone = process_record(rec)

            # Validation
            assert len(note_pitch) == len(rec["phones"]), "length mismatch"
            assert all(0 <= p <= 127 for p in note_pitch), "pitch out of range"
            assert all(d > 0 for d in note_dur), "zero note_dur"
            for j in range(1, len(note_to_phone)):
                assert note_to_phone[j] >= note_to_phone[j - 1], "non-monotonic note_to_phone"

            rec["note_pitch"] = note_pitch
            rec["note_dur"] = note_dur
            rec["note_to_phone"] = note_to_phone
            out_records.append(rec)
            ds_counts[rec["dataset"]] += 1

        except Exception as e:
            ds_errors[rec["dataset"]] += 1
            if len(issues) < 20:
                issues.append(f"{rec['dataset']}/{rec['seg_id']}: {e}")

        if (i + 1) % 20000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  {i+1:>7d} / {len(records)}  ({rate:.0f} rec/s)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({len(records)/elapsed:.0f} rec/s)")

    # Write output
    with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if skipped:
        print(f"Skipped {skipped} records from excluded datasets: {EXCLUDED_DATASETS}")
    print(f"\nOutput: {len(out_records)} records -> {MANIFEST_OUT}")
    print(f"  ({os.path.getsize(MANIFEST_OUT) / 1024 / 1024:.1f} MB)")

    # Per-dataset stats
    print(f"\n{'Dataset':20s} {'OK':>7s} {'Errors':>7s}")
    all_ds = sorted(set(list(ds_counts.keys()) + list(ds_errors.keys())))
    for ds in all_ds:
        print(f"  {ds:20s} {ds_counts[ds]:>7d} {ds_errors.get(ds,0):>7d}")

    if issues:
        print(f"\nFirst {len(issues)} errors:")
        for iss in issues:
            print(f"  {iss}")

    # Note pitch distribution for MIDI datasets
    print(f"\n=== NOTE PITCH STATS (MIDI datasets only) ===")
    midi_ds = set(ACE_SPLIT_DIR) | {"m4singer"} | set(GT_LANG_DIR) | set(JA_MIDI_DATASETS)
    pitch_counter = Counter()
    rest_count = 0
    for rec in out_records:
        if rec["dataset"] in midi_ds:
            for p in rec["note_pitch"]:
                if p == 0:
                    rest_count += 1
                else:
                    pitch_counter[p] += 1

    if pitch_counter:
        pitches = sorted(pitch_counter.keys())
        print(f"  Range: MIDI {pitches[0]}-{pitches[-1]}")
        print(f"  Rest (0): {rest_count}")
        top5 = pitch_counter.most_common(5)
        print(f"  Top 5: {', '.join(f'{p}({n})' for p, n in top5)}")

    # F0-inferred stats
    print(f"\n=== F0-INFERRED STATS ===")
    f0_pitches = Counter()
    f0_rest = 0
    for rec in out_records:
        if rec["dataset"] in F0_DATASETS:
            for p in rec["note_pitch"]:
                if p == 0:
                    f0_rest += 1
                else:
                    f0_pitches[p] += 1
    if f0_pitches:
        fps = sorted(f0_pitches.keys())
        print(f"  Range: MIDI {fps[0]}-{fps[-1]}")
        print(f"  Rest (0): {f0_rest}")
        top5 = f0_pitches.most_common(5)
        print(f"  Top 5: {', '.join(f'{p}({n})' for p, n in top5)}")


if __name__ == "__main__":
    main()
