# =============================================================================
# File: src/train.py
# Purpose: Stage‑2 training loop for the Diff‑Foley Latent Diffusion Model
# -----------------------------------------------------------------------------
# ‣ Loads a pre‑trained CAVP video encoder (frozen), a frozen EnCodec audio codec
#   and a fresh UNet‑based diffusion model with video cross‑attention.
# ‣ Trains only the UNet (and tiny code‑token projection) to predict noise in
#   the latent space.
# ‣ Logs running losses, saves checkpoints and an EMA of the UNet weights.
# =============================================================================

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from tqdm import tqdm

from utils.data_loader import make_dataloader
from models.latent_diffusion import LatentDiffusion
from models.video_encoder import CAVP
from models.audio_autoencoder import EncodecWrapper
from models.sampler import DDIMSampler

# ────────────────────────────────────────────────────────────────────────────
#  UNet factory (diffusers) – kept here so train.py is self‑contained
# ────────────────────────────────────────────────────────────────────────────
try:
    from diffusers import UNet2DConditionModel
except ImportError:
    raise RuntimeError("diffusers missing →  add diffusers>=0.25 to requirements.txt")


def build_unet(in_channels: int, model_channels: int = 320) -> nn.Module:
    """Light wrapper that mirrors Diff‑Foley’s UNet geometry."""
    return UNet2DConditionModel(
        sample_size=(1, 32),                 # (height, width) – height is dummy (1)
        in_channels=in_channels,             # 8 for EnCodec‑24kHz
        out_channels=in_channels,            # predict ε with same channels
        layers_per_block=2,
        block_out_channels=(model_channels, model_channels * 2, model_channels * 4),
        down_block_types=(
            "DownBlock2D",      # 1/2 W
            "AttnDownBlock2D",  # 1/4 W + self‑attention
            "AttnDownBlock2D",  # 1/8 W
        ),
        up_block_types=(
            "AttnUpBlock2D",    # 1/4 W
            "AttnUpBlock2D",    # 1/2 W
            "UpBlock2D",        # full W
        ),
        cross_attention_dim=512,            # matches CAVP token size
    )


# ────────────────────────────────────────────────────────────────────────────
#  Exponential‑moving‑average helper
# ────────────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])


# ────────────────────────────────────────────────────────────────────────────
#  Training loop
# ────────────────────────────────────────────────────────────────────────────

def train_loop(cfg: OmegaConf) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1 ─ Data
    train_loader = make_dataloader(cfg.data, split="train")

    # 2 ─ Models (codec + video encoder are frozen internally)
    codec = EncodecWrapper(target_sr=cfg.audio.sample_rate).to(device).eval()
    latent_channels = codec.code_embed.embedding_dim  # usually 8

    unet = build_unet(in_channels=latent_channels, model_channels=cfg.model.base_width)
    unet.to(device)

    cavp = CAVP().to(device).eval()  # weights already loaded in its __init__

    ldm = LatentDiffusion(
        unet=unet,
        timesteps=cfg.diffusion.timesteps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        guidance_prob=cfg.diffusion.guidance_p,
        latent_width=cfg.data.latent_width,
        target_sr=cfg.audio.sample_rate,
        device=device,
    ).to(device)

    # 3 ─ Optimiser & EMA
    optimiser = optim.AdamW(ldm.unet.parameters(), lr=cfg.optim.lr, weight_decay=1e-2)
    ema = EMA(ldm.unet, decay=cfg.optim.ema_decay)
    """"
    # 4 ─ Optionally resume
    start_step = 0
    ckpt_dir = Path(cfg.training.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / "last.pt"
    if latest_ckpt.exists():
        state = torch.load(latest_ckpt, map_location="cpu")
        ldm.load_state_dict(state["model"], strict=False)
        optimiser.load_state_dict(state["optim"])
        ema.shadow = state["ema"]
        start_step = state["step"] + 1
        print(f"▶ Resumed from step {start_step}.")
    """"
    # 5 ─ Training
    global_step = start_step
    ldm.train()
    for epoch in range(cfg.training.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            wav   = batch["audio"].to(device)   # (B, T)
            video = batch["video"].to(device)   # (B, C, F, H, W)

            loss = ldm(wav, video)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            ema.update(ldm.unet)

            pbar.set_postfix(loss=loss.item())

            # ─ Checkpoint
            if global_step % cfg.training.ckpt_every == 0:
                ckpt = {
                    "model": ldm.state_dict(),
                    "optim": optimiser.state_dict(),
                    "ema":   ema.shadow,
                    "step":  global_step,
                }
                torch.save(ckpt, ckpt_dir / f"step{global_step}.pt")
                torch.save(ckpt, latest_ckpt)

            global_step += 1

    # Save EMA‑smoothed weights at the very end
    ema.copy_to(ldm.unet)
    torch.save(ldm.state_dict(), ckpt_dir / "ldm_ema_final.pt")
    print("✅ Training finished – EMA model saved.")


# ────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Diff‑Foley LDM (Stage‑2)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    train_loop(cfg)


if __name__ == "__main__":
    main()
