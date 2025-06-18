# =============================================================================
# File: src/train_LDM.py
# Purpose: Stage-2 training loop for the Diff-Foley Latent Diffusion Model
# -----------------------------------------------------------------------------
# ‣ Loads a pre-trained CAVP video encoder (frozen), a frozen EnCodec audio codec
#   and a fresh UNet-based diffusion model with video cross-attention.
# ‣ Trains only the UNet (and tiny code-token projection) to predict noise in
#   the latent space.
# ‣ Logs running losses, saves checkpoints and an EMA of the UNet weights.
# ‣ Uses VidSpectroDataset to load (mel-spectrogram, video) pairs.
# =============================================================================

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import VidSpectroDataset
from models.latent_diffusion import LatentDiffusion
from models.cavp_encoder import CAVP, CAVP_VideoOnly
from models.audio_autoencoder import EncodecWrapper
from models.sampler import DPMSolverSampler

from torch.amp import autocast

# ────────────────────────────────────────────────────────────────────────────
#  UNet factory (diffusers) – kept here so train_LDM.py is self-contained
# ────────────────────────────────────────────────────────────────────────────

from diffusers import UNet2DConditionModel


def build_unet(
        in_channels: int,
        model_channels: int = 320,
        latent_w: int = 64,
        cross_attn_dim: int = 512
    ) -> nn.Module:
    """Fabrique un UNet de diffusion audio‐latent
    entièrement paramétrable en largeur temporelle
    et en dimension de cross‐attention vidéo."""
    return UNet2DConditionModel(
        sample_size           = (latent_w, latent_w),     # (H, W)
        in_channels           = in_channels,
        out_channels          = in_channels,
        layers_per_block      = 2,
        block_out_channels    = (model_channels,
                                  model_channels * 2,
                                  model_channels * 4),
        down_block_types      = ("DownBlock2D",
                                 "AttnDownBlock2D",
                                 "AttnDownBlock2D"),
        up_block_types        = ("AttnUpBlock2D",
                                 "AttnUpBlock2D",
                                 "UpBlock2D"),
        cross_attention_dim   = cross_attn_dim
    )




# ────────────────────────────────────────────────────────────────────────────
#  Exponential-moving-average helper
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
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # 1 ─ Dataset & DataLoader using VidSpectroDataset
    dataset = VidSpectroDataset(data_path=cfg.data.path, device=device)
    train_loader = DataLoader(
        dataset,
        batch_size   = cfg.data.batch_size,
        shuffle      = True,
        num_workers  = cfg.data.num_workers,
        pin_memory   = True,
        drop_last    = True,
        collate_fn   = VidSpectroDataset.collate_fn
    )

    # 2 ─ Models (codec + video encoder are frozen internally)
    codec = EncodecWrapper().to(device).eval()
    latent_channels = codec.latent_dim

    unet = build_unet(
        in_channels    = latent_channels,
        model_channels = cfg.model.base_width,
        latent_w       = cfg.data.latent_width,
        cross_attn_dim = 512
    ).to(device)


    cavp = CAVP_VideoOnly(cfg.cavp.checkpoint).to(device)

    ldm = LatentDiffusion(
        codec         = codec,
        unet          = unet,
        cavp          = cavp,
        timesteps     = cfg.diffusion.timesteps,
        beta_start    = cfg.diffusion.beta_start,
        beta_end      = cfg.diffusion.beta_end,
        guidance_prob = cfg.diffusion.guidance_p,
        device        = device,
    ).to(device)

    # 3 ─ Optimiser & EMA
    optimiser = optim.AdamW(ldm.unet.parameters(), lr=cfg.optim.lr, weight_decay=1e-2)
    ema = EMA(ldm.unet, decay=cfg.optim.ema_decay)

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

    # 5 ─ Training
    ldm.train()
    global_step = start_step
    total_steps = cfg.training.total_steps

    pbar = tqdm(total=total_steps, initial=global_step, unit="step")
    with autocast(device_type=device_str):
        while global_step < total_steps:
            for i, data in enumerate(train_loader):
                # batch["audio"]: (B, 1, n_mels, T)   (mel-spectrogram)
                # batch["video"]: (B, 3, F, H, W)      (video frames)
                spec  = data["audio"].to(device)
                video = data["video"].to(device)

                #new audio encoder accepts spec as input
                loss = ldm(spec, video)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                ema.update(ldm.unet)

                pbar.set_description(f"loss={loss.item():.4f}")
                pbar.update(1)
                print(" ")

                # ─ Checkpoint
                if global_step != 0 and global_step % cfg.training.ckpt_every == 0:
                    ckpt = {
                        "model": ldm.state_dict(),
                        "optim": optimiser.state_dict(),
                        "ema":   ema.shadow,
                        "step":  global_step,
                    }
                    torch.save(ckpt, ckpt_dir / f"step{global_step}.pt")
                    torch.save(ckpt, latest_ckpt)

                global_step += 1

    # Save EMA-smoothed weights at the very end
    ema.copy_to(ldm.unet)
    torch.save(ldm.state_dict(), ckpt_dir / "ldm_ema_final.pt")
    print("✅ Training finished – EMA model saved.")


# ────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Diff-Foley LDM (Stage-2)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    train_loop(cfg)


if __name__ == "__main__":
    main()
