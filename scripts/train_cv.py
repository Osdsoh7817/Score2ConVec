"""Train ScoreToCV (S58 FINAL clean cv model) — det_head only.

Objective = det_loss (cv MSE in normalized space) + pos_aux. No FM, no F0, no stages.
Quality signal = per-lang det_floor (conditional-mean cos to GT); EAR is the judge.
  py -3.10 scripts/train_cv.py --config configs/model_cv_final.yaml --run-name cvfinal
"""
import argparse, logging, math, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model.score2cv import ScoreToCV
from src.training.dataset import S2CVDataset, collate_fn


def cosine_warmup(opt, warmup, total, min_ratio=0.05):
    def f(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return max(min_ratio, 0.5 * (1 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def pos_aux_loss(out, mask, w):
    pp, pt = out.get("pos_pred"), out.get("pos_targets")
    if w <= 0 or pp is None or pt is None or not mask.any():
        return torch.tensor(0.0, device=mask.device)
    T = mask.shape[1]
    return F.l1_loss(pp[:, :T][mask], pt[:, :T][mask].clamp(0, 1))


@torch.no_grad()
def validate(model, val_loader, device, use_amp, pos_aux_w, sp_id=3, ap_id=4):
    model.eval()
    det_sum = pos_sum = 0.0; nb = 0
    pld_s = defaultdict(float); pld_n = defaultdict(int); dc_s = 0.0; dc_n = 0
    for batch in val_loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = model(phonemes=batch["phonemes"], note_pitch=batch["note_pitch"], phone_dur=batch["phone_dur"],
                        note_dur=batch["note_dur"], note_to_phone=batch["note_to_phone"],
                        speaker_id=batch["speaker_id"], lang_id=batch["lang_id"], phone_mask=batch["phone_mask"],
                        technique=batch["technique"])
            det = model.det_loss(out["frame_hidden"], batch["contentvec"], out["frame_mask"])
        det_sum += det.item(); pos_sum += float(pos_aux_loss(out, out["frame_mask"], pos_aux_w)); nb += 1
        mask = out["frame_mask"]; T = mask.shape[1]
        cmask = mask & (batch["f0"][:, :T] > 30) & (batch["frame_phone_ids"][:, :T] != sp_id) \
                & (batch["frame_phone_ids"][:, :T] != ap_id)
        mu = model.det_head(out["frame_hidden"]).float() * model.cv_std + model.cv_mean
        cosd = F.cosine_similarity(mu, batch["contentvec"][:, :T].float(), dim=-1)
        for i in range(mask.shape[0]):
            cmi = cmask[i]
            if cmi.any():
                dc = float(cosd[i][cmi].mean())
                pld_s[batch["lang"][i]] += dc; pld_n[batch["lang"][i]] += 1; dc_s += dc; dc_n += 1
    return {"det": det_sum / max(nb, 1), "pos_aux": pos_sum / max(nb, 1), "det_cos": dc_s / max(dc_n, 1),
            "per_lang_det_cos": {l: pld_s[l] / max(pld_n[l], 1) for l in sorted(pld_s)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model_cv_final.yaml")
    ap.add_argument("--resume", default=None); ap.add_argument("--run-name", default=None)
    ap.add_argument("--smoke-test", action="store_true")
    A = ap.parse_args()
    cfg = yaml.safe_load(open(A.config, encoding="utf-8")); tc = cfg["training"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = tc["fp16"] and dev.type == "cuda"
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    run = A.run_name or f"cvfinal_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path("runs") / run; ckpt = log_dir / "checkpoints"; ckpt.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "train.log", encoding="utf-8")])
    log = logging.getLogger("train_cv")

    root = Path("processed")
    aug = root / "splits" / "train_aug.jsonl"; speech = root / "splits" / "train_speech.jsonl"
    use_aug = tc.get("use_aug", False); use_speech = tc.get("use_speech", False)
    train_ds = S2CVDataset(str(root), str(root / "splits" / tc["train_split"]), max_frames=tc["max_frames"],
                           max_phones=100, lang_weights=tc.get("lang_weights"), augment=use_aug,
                           aug_split_file=str(aug) if (aug.exists() and use_aug) else None,
                           speech_split_file=str(speech) if (speech.exists() and use_speech) else None,
                           npz_root=cfg.get("npz_root"))
    val_ds = S2CVDataset(str(root), str(root / "splits" / tc.get("val_split", "val.jsonl")),
                         max_frames=tc["max_frames"], max_phones=100, npz_root=cfg.get("npz_root"))
    log.info(f"Train {len(train_ds):,} | Val {len(val_ds):,} | {train_ds.stats()['total_hours']:.1f}h "
             f"per-lang {train_ds.stats()['per_lang']}")
    sampler = train_ds.get_sampler() if tc.get("weighted_sampling") else None
    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], sampler=sampler, shuffle=(sampler is None),
                              collate_fn=collate_fn, num_workers=tc["num_workers"], pin_memory=True, drop_last=True,
                              persistent_workers=tc.get("persistent_workers", False),
                              prefetch_factor=6 if tc["num_workers"] > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False, collate_fn=collate_fn,
                            num_workers=2, pin_memory=True)

    model = ScoreToCV(cfg).to(dev)
    pos_w = tc.get("pos_aux_weight", 0.0)
    log.info(f"Model: {model.count_parameters():,} TRAINABLE params (ScoreToCV clean det)")
    opt = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"], betas=(0.9, 0.98), weight_decay=0.01)
    max_steps = tc["max_steps"]; sched = cosine_warmup(opt, tc["warmup_steps"], max_steps)

    step = 0; best = float("inf")
    if A.resume:
        ck = torch.load(A.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        step = ck["global_step"]; best = ck.get("best_val_loss", float("inf")); log.info(f"Resumed step {step}")
    yaml.dump(cfg, open(log_dir / "config.yaml", "w"), default_flow_style=False)

    val_interval, save_interval = 2000, 5000
    model.train(); t0 = time.time(); running = defaultdict(float); nacc = 0; stop = False
    log.info(f"Training: max_steps={max_steps} batch={tc['batch_size']} lr={tc['learning_rate']} amp={use_amp}")
    while step < max_steps and not stop:
        for batch in train_loader:
            if step >= max_steps:
                break
            batch = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                out = model(phonemes=batch["phonemes"], note_pitch=batch["note_pitch"], phone_dur=batch["phone_dur"],
                            note_dur=batch["note_dur"], note_to_phone=batch["note_to_phone"],
                            speaker_id=batch["speaker_id"], lang_id=batch["lang_id"], phone_mask=batch["phone_mask"],
                            technique=batch["technique"])
                det = model.det_loss(out["frame_hidden"], batch["contentvec"], out["frame_mask"])
                pos = pos_aux_loss(out, out["frame_mask"], pos_w)
                loss = det + pos_w * pos
            if not torch.isfinite(loss):
                log.warning(f"non-finite loss @ {step+1} — skip"); opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); loss.backward()
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"]).item()
            opt.step(); sched.step(); step += 1
            running["det"] += det.item(); running["pos"] += float(pos); running["total"] += loss.item(); nacc += 1

            if A.smoke_test:
                log.info(f"Smoke train OK: det={det.item():.4f} pos={float(pos):.4f} gnorm={gnorm:.2f}")
                if dev.type == "cuda":
                    log.info(f"Peak GPU: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
                vl = validate(model, val_loader, dev, use_amp, pos_w)
                log.info(f"Smoke val OK: det={vl['det']:.4f} det_floor={vl['det_cos']:.3f}")
                log.info("Smoke test PASSED"); return

            if step % 100 == 0:
                a = {k: v / nacc for k, v in running.items()}; lr = opt.param_groups[0]["lr"]
                sps = step / (time.time() - t0)
                log.info(f"step={step:>6d} det={a['det']:.4f} pos={a['pos']:.4f} gnorm={gnorm:.2f} "
                         f"lr={lr:.2e} [{sps:.1f} it/s, ETA {(max_steps-step)/sps/3600:.1f}h]")
                running = defaultdict(float); nacc = 0
            if step % val_interval == 0:
                pld = (vl := validate(model, val_loader, dev, use_amp, pos_w)).pop("per_lang_det_cos", {})
                log.info(f"[VAL] step={step:>6d} det={vl['det']:.4f} det_floor={vl['det_cos']:.3f} pos={vl['pos_aux']:.4f}")
                log.info("[VAL] per_lang det_floor: " + " ".join(f"{l}={v:.3f}" for l, v in pld.items()))
                bestm = 1.0 - vl["det_cos"]
                if bestm < best:
                    best = bestm
                    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                                "scheduler": sched.state_dict(), "global_step": step, "best_val_loss": best,
                                "config": cfg}, ckpt / "best.pt")
                    log.info(f"  -> new best det_floor={vl['det_cos']:.3f}")
                model.train()
            if step % save_interval == 0:
                torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                            "global_step": step, "best_val_loss": best, "config": cfg}, ckpt / f"step_{step}.pt")
    torch.save({"model": model.state_dict(), "global_step": step, "config": cfg}, ckpt / "final.pt")
    log.info(f"Done in {(time.time()-t0)/3600:.1f}h. Best 1-det_floor={best:.4f}")


if __name__ == "__main__":
    main()
