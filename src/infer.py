# =============================================================================
# File: src/infer.py
# Role: Generate Foley audio for a silent video
# -----------------------------------------------------------------------------
# 1. Load checkpoint, frozen encoders, and sampler.
# 2. Extract CAVP tokens from input video.
# 3. Sample noise z_T → denoise with DDIM/DPM-Solver + CFG.
# 4. Decode latent → mel → WAV, then save (and optionally mux to MP4).
# =============================================================================

#!/usr/bin/env python3
"""video‑to‑audio inference script

Runs the two‑stage Video➜Audio latent diffusion model on an input video and
produces a Foley soundtrack. Optionally muxes the generated audio back under
( a copy of ) the original clip using *ffmpeg*.

Example
-------
python infer.py \
    --video data/demo.mp4 \
    --ldm_ckpt checkpoints/ldm/ldm-step4500.pt \
    --cavp_ckpt checkpoints/cavp/cavp-7500steps.pt \
    --out_audio demo_foley.wav \
    --out_video demo_foley.mp4  --mux

All command‑line options are documented with ``--help``.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from typing import Tuple

import torch
import torchaudio
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from torchvision.io import read_video

# --- repository internal imports (relative to repo root) -------------------
from models.audio_autoencoder import EncodecWrapper
from models.cavp_encoder import CAVP_VideoOnly
from models.latent_diffusion import LatentDiffusion
from models.sampler import DPMSolverSampler
from train_LDM import build_unet
from utils.dataset import (
    F_MAX,
    F_MIN,
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
)

# ---------------------------------------------------------------------------
#   helpers
# ---------------------------------------------------------------------------

def video_to_tensor(
    path: str | pathlib.Path,
    fps: int = 4,
    frames: int = 40,
    size: int = 224,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Load an arbitrary short clip and spit out (1,3,T,H,W) *float32* in [0,1]."""

    vid, _, meta = read_video(str(path), pts_unit="sec")  # (T,H,W,C) uint8 CPU
    stride = max(1, int(meta["video_fps"] // fps)) if meta.get("video_fps") else 1
    vid = vid[::stride][:frames]

    # left‑pad if shorter than the fixed training length
    if vid.shape[0] < frames:
        pad = torch.zeros(frames - vid.shape[0], *vid.shape[1:], dtype=vid.dtype)
        vid = torch.cat([vid, pad])

    vid = TF.resize(vid.permute(0, 3, 1, 2), size) / 255.0  # (T,3,H,W)
    vid = vid.permute(1, 0, 2, 3).unsqueeze(0)  # (1,3,T,H,W)
    return vid.to(device, non_blocking=True)


def build_model(
    ldm_ckpt: str | pathlib.Path,
    cavp_ckpt: str | pathlib.Path,
    cfg: "OmegaConf | dict",
    device: torch.device | str,
) -> Tuple[LatentDiffusion, DPMSolverSampler]:
    """Create EnCodec, CAVP, UNet stack and load weights."""

    device = torch.device(device)

    codec = EncodecWrapper(device=device).eval()
    cavp = CAVP_VideoOnly(str(cavp_ckpt)).to(device).eval()

    unet = build_unet(
        in_channels=codec.latent_dim,
        model_channels=256,
        latent_w=64,
        cross_attn_dim=512,
    ).to(device)

    ldm = LatentDiffusion(
        codec=codec,
        unet=unet,
        cavp=cavp,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        guidance_prob=0.0,
        device=device,
    ).to(device)

    ckpt = torch.load(ldm_ckpt, map_location="cpu")
    missing, unexpected = ldm.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        print(f"[infer] ⚠️  state‑dict loaded with missing={len(missing)} unexpected={len(unexpected)}", file=sys.stderr)

    sampler = DPMSolverSampler(ldm)
    ldm.eval()
    return ldm, sampler


def mel_to_waveform(mel_db: torch.Tensor) -> torch.Tensor:
    """Quick & dirty Griffin‑Lim: (1,1,128,T) log‑mel → mono waveform."""
    mel_lin = 10.0 ** (mel_db.squeeze(0) / 10.0)

    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=N_FFT // 2 + 1,
        n_mels=128,
        sample_rate=SAMPLE_RATE,
        f_min=F_MIN,
        f_max=F_MAX,
    ).to(mel_lin.device)
    spec = inv_mel(mel_lin)

    wave = torchaudio.functional.griffinlim(
        spec,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=torch.hann_window(N_FFT).to(mel_lin.device),
    )
    return wave.cpu()


# ---------------------------------------------------------------------------
#   main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required
    p.add_argument("--video", type=str, required=True, help="input video file (any format ffmpeg can read)")
    p.add_argument("--ldm_ckpt", type=str, required=True, help="path to UNet / LDM checkpoint")
    p.add_argument("--cavp_ckpt", type=str, required=True, help="path to frozen CAVP checkpoint")

    # optional paths
    p.add_argument("--out_audio", type=str, default="foley.wav", help="output .wav filename")
    p.add_argument("--out_video", type=str, default="foley_out.mp4", help="muxed output video (only if --mux)")

    # generation params
    p.add_argument("--config", type=str, default="configs/infer.yaml", help="YAML with defaults for steps/cfg_scale")
    p.add_argument("--seconds", type=float, default=4.0, help="length of audio to generate (sec)")
    p.add_argument("--steps", type=int, default=None, help="diffusion steps (None → take from YAML)")
    p.add_argument("--guidance", type=float, default=None, help="CFG scale (None → take from YAML)")

    # misc
    p.add_argument("--device", type=str, default="cuda", help="cuda | cpu | cuda:1 …")
    p.add_argument("--mux", action="store_true", help="mux generated audio back with original video")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = OmegaConf.load(args.config)
    steps = int(args.steps if args.steps is not None else cfg.get("steps", 50))
    guidance = float(args.guidance if args.guidance is not None else cfg.get("cfg_scale", 3.0))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------------
    print("[infer] building model …")
    ldm, sampler = build_model(args.ldm_ckpt, args.cavp_ckpt, cfg, device)

    # ---------------------------------------------------------------------
    print("[infer] encoding video …")
    frames = video_to_tensor(args.video, device=device)

    # ---------------------------------------------------------------------
    print(f"[infer] sampling {args.seconds}s / {steps} steps  (CFG={guidance}) …")
    with torch.no_grad():
        mel_db = ldm.generate(
            frames,
            seconds=args.seconds,
            steps=steps,
            guidance=guidance,
            sampler=sampler,
        )  # (1,1,128,T)

    print("[infer] mel → waveform …")
    waveform = mel_to_waveform(mel_db)
    torchaudio.save(args.out_audio, waveform.unsqueeze(0), SAMPLE_RATE)
    print("[infer] ✅  wrote", args.out_audio)

    # ------------------------------------------------------------------
    if args.mux:
        print("[infer] ffmpeg mux …")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(args.video),
            "-i",
            str(args.out_audio),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(args.out_video),
        ]
        subprocess.run(cmd, check=True)
        print("[infer] 🎬  wrote", args.out_video)

    print("[infer] done ✔︎")


if __name__ == "__main__":
    main()
