"""Train ScoreToF0 (S58) — generative F0 (cents-deviation FM) + voicing head.

cents target = 1200*log2(GT_f0 / note_hz), cleaned (voiced+pitched, |cents|<800, no octave-jump).
voicing target = (GT_f0 > 30) — SP-RELAXED (NO forced-0 at SP; held-into-rest sustains stay voiced).
  py -3.10 scripts/train_f0.py --config configs/model_f0.yaml --run-name f0final
"""
import argparse, logging, math, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model.score2f0 import ScoreToF0
from src.training.dataset import S2CVDataset, collate_fn


def cosine_warmup(opt, warmup, total, min_ratio=0.05):
    def f(s):
        if s < warmup:
            return s / max(1, warmup)
        p = (s - warmup) / max(1, total - warmup)
        return max(min_ratio, 0.5 * (1 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def octave_jump(f0, k=9, thresh=0.583):
    """[B,T] bool: voiced frames > thresh octaves from local voiced-median log-f0 (RMVPE doubling/halving)."""
    voiced = f0 > 30
    lf = torch.log2(f0.clamp(min=1.0))
    pad = k // 2
    win = F.pad(lf, (pad, pad)).unfold(1, k, 1)
    winv = F.pad(voiced.float(), (pad, pad)).unfold(1, k, 1)
    med = win.masked_fill(winv < 0.5, float("nan")).nanmedian(dim=-1).values
    return voiced & torch.isfinite(med) & ((lf - med).abs() > thresh)


def cents_and_masks(batch, frame_mask, T):
    """cents target [B,T] + clean-fm mask + voicing target (SP-relaxed)."""
    f0 = batch["f0"][:, :T]
    midi = batch["frame_note_pitch"][:, :T].float()
    midi_hz = 440.0 * 2.0 ** ((midi - 69.0) / 12.0)
    valid = (f0 > 30) & (midi > 0)
    cents = torch.where(valid, 1200.0 * torch.log2(f0.clamp(min=1e-6) / midi_hz.clamp(min=1e-6)),
                        torch.zeros_like(f0))
    clean = valid & (cents.abs() < 800) & ~octave_jump(f0) & frame_mask
    voicing_tgt = (f0 > 30).float()                                   # SP-RELAXED (no SP override)
    return cents, clean, voicing_tgt


@torch.no_grad()
def validate(model, val_loader, device, use_amp, voi_w):
    model.eval()
    det_sum = fm_sum = voi_sum = 0.0; nb = 0
    mae_sum = 0.0; mae_n = 0; vacc_sum = 0.0; vacc_n = 0
    pl_mae = defaultdict(float); pl_n = defaultdict(int)
    for bi, batch in enumerate(val_loader):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            fh, fmask, midi_f = model.encode(
                phonemes=batch["phonemes"], note_pitch=batch["note_pitch"], phone_dur=batch["phone_dur"],
                note_dur=batch["note_dur"], note_to_phone=batch["note_to_phone"], speaker_id=batch["speaker_id"],
                lang_id=batch["lang_id"], phone_mask=batch["phone_mask"], technique=batch["technique"])
            T = fh.shape[1]
            cents, clean, vtgt = cents_and_masks(batch, fmask, T)
            det = model.det_loss(fh, cents, clean)
            fm = model.fm_loss(fh, cents, clean)
            vlogit = model.voicing_logit(fh)
            voi = F.binary_cross_entropy_with_logits(vlogit[fmask], vtgt[:, :T][fmask])
        det_sum += det.item(); fm_sum += fm.item(); voi_sum += voi.item(); nb += 1
        # sampled cents MAE (one sample) + voicing accuracy, on first few val batches (sampling is slow)
        if bi < 4:
            cents_s = model.sample_cents(fh.float(), fmask).float()
            err = (cents_s - cents).abs()
            for i in range(fh.shape[0]):
                cm = clean[i]
                if cm.any():
                    m = float(err[i][cm].mean()); mae_sum += m; mae_n += 1
                    pl_mae[batch["lang"][i]] += m; pl_n[batch["lang"][i]] += 1
            vpred = (torch.sigmoid(vlogit) > 0.5)
            vacc_sum += float((vpred[fmask] == (vtgt[:, :T][fmask] > 0.5)).float().mean()); vacc_n += 1
    return {"det": det_sum / max(nb, 1), "fm": fm_sum / max(nb, 1), "voi": voi_sum / max(nb, 1),
            "cents_mae": mae_sum / max(mae_n, 1), "voi_acc": vacc_sum / max(vacc_n, 1),
            "per_lang_mae": {l: pl_mae[l] / max(pl_n[l], 1) for l in sorted(pl_mae)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model_f0.yaml")
    ap.add_argument("--resume", default=None); ap.add_argument("--run-name", default=None)
    ap.add_argument("--smoke-test", action="store_true")
    A = ap.parse_args()
    cfg = yaml.safe_load(open(A.config, encoding="utf-8")); tc = cfg["training"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = tc["fp16"] and dev.type == "cuda"
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    run = A.run_name or f"f0_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path("runs") / run; ckpt = log_dir / "checkpoints"; ckpt.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "train.log", encoding="utf-8")])
    log = logging.getLogger("train_f0")

    root = Path("processed")
    train_ds = S2CVDataset(str(root), str(root / "splits" / tc["train_split"]), max_frames=tc["max_frames"],
                           max_phones=100, lang_weights=tc.get("lang_weights"), augment=tc.get("use_aug", False))
    val_ds = S2CVDataset(str(root), str(root / "splits" / tc.get("val_split", "val.jsonl")),
                         max_frames=tc["max_frames"], max_phones=100)
    log.info(f"Train {len(train_ds):,} | Val {len(val_ds):,} | {train_ds.stats()['total_hours']:.1f}h")
    sampler = train_ds.get_sampler() if tc.get("weighted_sampling") else None
    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], sampler=sampler, shuffle=(sampler is None),
                              collate_fn=collate_fn, num_workers=tc["num_workers"], pin_memory=True, drop_last=True,
                              prefetch_factor=6 if tc["num_workers"] > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = ScoreToF0(cfg).to(dev)
    voi_w = tc.get("voicing_weight", 0.5)
    log.info(f"Model: {model.count_parameters():,} TRAINABLE params (ScoreToF0)")
    opt = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"], betas=(0.9, 0.98), weight_decay=0.01)
    max_steps = tc["max_steps"]; sched = cosine_warmup(opt, tc["warmup_steps"], max_steps)
    step = 0; best = float("inf")
    if A.resume:
        ck = torch.load(A.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        step = ck["global_step"]; best = ck.get("best_val_loss", float("inf")); log.info(f"Resumed step {step}")
    yaml.dump(cfg, open(log_dir / "config.yaml", "w"), default_flow_style=False)

    val_interval, save_interval = 2000, 5000
    model.train(); t0 = time.time(); running = defaultdict(float); nacc = 0
    log.info(f"Training: max_steps={max_steps} batch={tc['batch_size']} lr={tc['learning_rate']} voi_w={voi_w} amp={use_amp}")
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            batch = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                fh, fmask, midi_f = model.encode(
                    phonemes=batch["phonemes"], note_pitch=batch["note_pitch"], phone_dur=batch["phone_dur"],
                    note_dur=batch["note_dur"], note_to_phone=batch["note_to_phone"], speaker_id=batch["speaker_id"],
                    lang_id=batch["lang_id"], phone_mask=batch["phone_mask"], technique=batch["technique"])
                T = fh.shape[1]
                cents, clean, vtgt = cents_and_masks(batch, fmask, T)
                det = model.det_loss(fh, cents, clean)
                fm = model.fm_loss(fh, cents, clean)
                vlogit = model.voicing_logit(fh)
                voi = F.binary_cross_entropy_with_logits(vlogit[fmask], vtgt[:, :T][fmask])
                loss = det + fm + voi_w * voi
            if not torch.isfinite(loss):
                log.warning(f"non-finite @ {step+1} skip"); opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); loss.backward()
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"]).item()
            opt.step(); sched.step(); step += 1
            running["det"] += det.item(); running["fm"] += fm.item(); running["voi"] += voi.item(); nacc += 1
            if A.smoke_test:
                log.info(f"Smoke train OK: det={det.item():.4f} fm={fm.item():.4f} voi={voi.item():.4f} gnorm={gnorm:.2f}")
                if dev.type == "cuda":
                    log.info(f"Peak GPU: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
                vl = validate(model, val_loader, dev, use_amp, voi_w)
                log.info(f"Smoke val OK: det={vl['det']:.4f} fm={vl['fm']:.4f} voi={vl['voi']:.4f} cents_mae={vl['cents_mae']:.1f} voi_acc={vl['voi_acc']:.3f}")
                log.info("Smoke test PASSED"); return
            if step % 100 == 0:
                a = {k: v / nacc for k, v in running.items()}; lr = opt.param_groups[0]["lr"]
                sps = step / (time.time() - t0)
                log.info(f"step={step:>6d} det={a['det']:.4f} fm={a['fm']:.4f} voi={a['voi']:.4f} gnorm={gnorm:.2f} lr={lr:.2e} "
                         f"[{sps:.1f} it/s, ETA {(max_steps-step)/sps/3600:.1f}h]")
                running = defaultdict(float); nacc = 0
            if step % val_interval == 0:
                plm = (vl := validate(model, val_loader, dev, use_amp, voi_w)).pop("per_lang_mae", {})
                log.info(f"[VAL] step={step:>6d} det={vl['det']:.4f} fm={vl['fm']:.4f} voi={vl['voi']:.4f} cents_mae={vl['cents_mae']:.1f} voi_acc={vl['voi_acc']:.3f}")
                log.info("[VAL] per_lang cents_mae: " + " ".join(f"{l}={v:.0f}" for l, v in plm.items()))
                m = vl["fm"]
                if m < best:
                    best = m
                    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                                "global_step": step, "best_val_loss": best, "config": cfg}, ckpt / "best.pt")
                    log.info(f"  -> new best fm={best:.4f}")
                model.train()
            if step % save_interval == 0:
                torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                            "global_step": step, "best_val_loss": best, "config": cfg}, ckpt / f"step_{step}.pt")
    torch.save({"model": model.state_dict(), "global_step": step, "config": cfg}, ckpt / "final.pt")
    log.info(f"Done in {(time.time()-t0)/3600:.1f}h. Best fm={best:.4f}")


if __name__ == "__main__":
    main()
