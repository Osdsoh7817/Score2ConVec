"""Build npz256 CORRECTLY = existing npz fields (patched phonemes / de-crammed dur / f0 / notes KEPT)
+ ONLY contentvec swapped to the frame-aligned 256-d (vec256l9).

The existing npz diverged from manifest_final (S46 dict patches, S57 de-cram, gtsinger seq drift), so a
manifest rebuild is WRONG. Instead, for EACH existing npz we LOCATE its cv768 inside per_utt_768 and take
the same-position slice of per_utt_256 (proven frame-aligned, cos=1.0). Every clip is exact-verified
(pad-tolerant: trailing edge-pad allowed). Writes processed/npz256; does NOT touch processed/npz.

Path A (fast): manifest hint (utt, frame_start). Path B (fallback): content index (anchor = max-norm frame).
"""
import sys, json, collections
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent.parent   # repo root
NPZ = R / "processed" / "npz"; OUT = R / "processed" / "npz256"
PU768 = R / "processed" / "contentvec"; PU256 = R / "processed" / "contentvec256"
MAN = R / "processed" / "manifest_final.jsonl"
LINEUP = ["m4singer", "mfa_french", "mfa_german", "mfa_italian", "mfa_spanish",
          "gtsinger_en", "kiritan", "itako", "natsume", "ofuton", "oniku", "pjs"]

man = collections.defaultdict(list)
for l in open(MAN, encoding="utf-8"):
    r = json.loads(l); man[r["dataset"]].append(r)


def verify_and_slice(pu7, pu2, off, cv768, T):
    """If pu7[off:off+T] matches cv768 (trailing edge-pad allowed), return the aligned pu2 slice padded to T."""
    if pu7 is None or pu2 is None or off < 0:
        return None
    avail = pu7.shape[0] - off
    if avail <= 0:
        return None
    m = min(T, avail)
    if not np.array_equal(pu7[off:off + m], cv768[:m]):
        return None
    if m < T:  # the rest of cv768 must be edge-pad (repeat of the last real frame)
        if not np.array_equal(cv768[m:], np.broadcast_to(cv768[m - 1], (T - m, cv768.shape[1]))):
            return None
    cv256 = pu2[off:off + m]
    # GUARANTEE exactly T frames. per_utt_256 can be SHORTER than per_utt_768 for the same utt
    # (alignment total_frames drifted between the 6/9 768-extract and the 6/25 256-extract), so
    # pu2[off:off+m] may come up short and the old `if m<T` pad missed it -> cv256.T<f0.T -> collate
    # crash. Any length delta is TAIL-only (extraction pads/truncates at the end), so edge-padding the
    # tail to T preserves frame alignment to the score/f0.
    if cv256.shape[0] < T:
        cv256 = np.pad(cv256, ((0, T - cv256.shape[0]), (0, 0)), mode="edge")
    elif cv256.shape[0] > T:
        cv256 = cv256[:T]
    return cv256.astype(np.float16)


def main():
    grand = collections.Counter(); fails_all = []
    for ds in LINEUP:
        files = sorted((NPZ / ds).glob("*.npz"))
        (OUT / ds).mkdir(parents=True, exist_ok=True)
        recs = man.get(ds, [])
        cache = {}

        def getpu(utt):
            if utt not in cache:
                if len(cache) > 120:
                    cache.clear()
                p7 = PU768 / ds / (utt + ".npy"); p2 = PU256 / ds / (utt + ".npy")
                cache[utt] = (np.load(p7) if p7.exists() else None, np.load(p2) if p2.exists() else None)
            return cache[utt]

        index = None

        def get_index():
            nonlocal index
            if index is None:
                index = collections.defaultdict(list)
                for p in (PU768 / ds).rglob("*.npy"):
                    utt = str(p.relative_to(PU768 / ds)).replace("\\", "/")[:-4]
                    arr = np.load(p)
                    for i in range(arr.shape[0]):
                        index[hash(arr[i].tobytes())].append((utt, i))
                print(f"  [{ds}] content index built: {len(index)} keys", flush=True)
            return index

        c = collections.Counter()
        for f in files:
            e = np.load(f, allow_pickle=True)
            cv768 = e["contentvec"]; T = cv768.shape[0]
            try:
                seq = int(f.stem)
            except ValueError:
                seq = -1
            cv256 = None; via = None
            # Path A: manifest hint
            if 0 <= seq < len(recs):
                rec = recs[seq]; pu7, pu2 = getpu(rec["utt_id"])
                cv256 = verify_and_slice(pu7, pu2, rec["frame_start"], cv768, T)
                if cv256 is not None:
                    via = "hint"
            # Path B: content index
            if cv256 is None:
                idx = get_index()
                k = int(np.argmax(np.linalg.norm(cv768.astype(np.float32), axis=1)))
                for (utt, i) in idx.get(hash(cv768[k].tobytes()), []):
                    pu7, pu2 = getpu(utt)
                    cv256 = verify_and_slice(pu7, pu2, i - k, cv768, T)
                    if cv256 is not None:
                        via = "index"; break
            if cv256 is None:
                c["FAIL"] += 1
                if len(fails_all) < 30:
                    fails_all.append(f"{ds}/{f.name} T={T}")
                continue
            out = {kk: e[kk] for kk in e.files if kk != "contentvec"}
            out["contentvec"] = cv256
            np.savez(OUT / ds / f.name, **out)
            c[via] += 1; c["ok"] += 1
        print(f"{ds:12s} files={len(files)} {dict(c)}", flush=True)
        grand.update(c)
    print(f"\nTOTAL: {dict(grand)}")
    print("0 FAILS — every clip exact-aligned" if grand["FAIL"] == 0 else f"FAILS ({grand['FAIL']}): {fails_all}")


if __name__ == "__main__":
    main()
