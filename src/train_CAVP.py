# =============================================================================
# File: src/train_CAVP.py
# Purpose: Contrastive Audio-Visual Pre‑training (CAVP) stage
# -----------------------------------------------------------------------------
# ‣ Trains the CAVP model (audio + video encoders with shared logit_scale)
#   to align 4‑second clips via semantic (extra‑video) and temporal (intra‑video)
#   contrastive losses, as in Diff‑Foley (Eq. 1 & 2).
# ‣ Writes two frozen checkpoints:  audio_encoder.pth  and  video_encoder.pth
#   to be reused (frozen) in Stage‑2 LDM training.
# =============================================================================

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from tqdm import tqdm

# from utils.data_loader import make_dataloader            # returns dict with keys: video, mel, video_id
from models.cavp_encoder import CAVP, CAVP_Loss         # model & loss share logit_scale

from torch.utils.data import DataLoader
from utils.dataset import VidSpectroDataset

from torch.cuda.amp import autocast

# ────────────────────────────────────────────────────────────────────────────
#  Training procedure
# ────────────────────────────────────────────────────────────────────────────

def train_cavp(cfg: OmegaConf) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1 ─ Data ----------------------------------------------------------------
    # loader = make_dataloader(cfg.data, split="train")  # should yield mel‑spectrograms, not raw wav

    # # pytorch dataloader support
    dataset = VidSpectroDataset(cfg.data.path, device=device)
    loader = DataLoader(dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, shuffle=True, collate_fn=VidSpectroDataset.collate_fn)

    # 2 ─ Model + Loss --------------------------------------------------------
    model = CAVP(feat_dim=cfg.model.feat_dim,
                 temperature=cfg.loss.temperature).to(device)

    model.audio_encoder.load_state_dict(torch.load("pretrained-models/cnn14-encoder.pth", map_location="cpu"))
    model.video_encoder.load_state_dict(torch.load("pretrained-models/resnet50-slowfast-encoder.pth", map_location="cpu"))

    criterion = CAVP_Loss(clip_num=cfg.loss.clip_num, lambda_=cfg.loss.lambda_).to(device)

    # 3 ─ Optimiser -----------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(),   # criterion has no standalone params
        lr=cfg.optim.lr,
    )

    # 4 ─ Checkpoint resume ---------------------------------------------------
    ckpt_dir = Path(cfg.training.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "last.pt"
    start_step = 0
    if latest.exists():
        state = torch.load(latest, map_location="cpu")
        model.load_state_dict(state["model"], strict=False)
        criterion.load_state_dict(state["loss"])
        optimizer.load_state_dict(state["optim"])
        start_step = state["step"] + 1
        print(f"▶ Resumed CAVP from step {start_step}.")

    # 5 ─ Training loop -------------------------------------------------------
    model.train()
    global_step = start_step
    total_steps = cfg.training.total_steps

    pbar = tqdm(total=total_steps, initial=global_step, unit="step")
    with autocast():
      while global_step < total_steps:
          for i, data in enumerate(loader):
              video = data["video"].to(device)          # (B, 3, F, H, W)
              mel   = data["audio"].to(device)            # (B, 1, n_mels, T)

              # Forward ------------------------------------------------------
              video_feats, video_mean_feats, audio_feats, audio_mean_feats, logit_scale = model(video, mel)      # order matches model.forward
              loss = criterion(video_feats, video_mean_feats, audio_feats, audio_mean_feats, logit_scale)

              # Back‑prop ----------------------------------------------------
              optimizer.zero_grad()
              loss.backward()
              nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_grad_norm)
              optimizer.step()

              # Logging & checkpoint ---------------------------------------
              pbar.set_description(f"loss={loss.item():.4f}")
              pbar.update(1)
              print("  ")

              if global_step > 0 and global_step % cfg.training.ckpt_every == 0:
                  ckpt = {
                      "model": model.state_dict(),
                      "loss": criterion.state_dict(),
                      "optim": optimizer.state_dict(),
                      "step": global_step,
                  }
                  torch.save(ckpt, latest)
                  torch.save(ckpt, ckpt_dir / f"step{global_step}.pt")

              global_step += 1
              if global_step >= total_steps:
                  break

    # 6 ─ Save final encoders -------------------------------------------------
    torch.save(model.audio_encoder.state_dict(), ckpt_dir / "audio_encoder.pth")
    torch.save(model.video_encoder.state_dict(), ckpt_dir / "video_encoder.pth")
    print("✅ CAVP training completed – encoders saved.")


# ────────────────────────────────────────────────────────────────────────────
#  CLI helper
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CAVP encoders (Stage‑1)")
    p.add_argument("--config", type=str, required=True, help="Path to YAML cfg")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    train_cavp(cfg)


if __name__ == "__main__":
    main()