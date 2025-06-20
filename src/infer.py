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
import os
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
    TARGET_FPS,
    TARGET_SIZE,
    FIXED_NUM_FRAMES,
    VidSpectroDataset,
)

# ---------------------------------------------------------------------------
#   helpers
# ---------------------------------------------------------------------------

def get_project_root() -> pathlib.Path:
    """Get the project root directory (parent of src/)."""
    return pathlib.Path(__file__).parent.parent

def process_video_for_inference(
    video_path: str | pathlib.Path,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Process video using the same logic as VidSpectroDataset.gen_vid()."""
    
    # Check if file exists
    video_path = pathlib.Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path.absolute()}")
    
    print(f"[infer] 📹 Loading video: {video_path}")
    
    try:
        # Use the same logic as dataset.py gen_vid()
        frames_data, _, meta = read_video(str(video_path), pts_unit="sec")
    except Exception as e:
        print(f"[infer] ❌ Failed to load video: {e}")
        print(f"[infer] 💡 Supported formats: .mp4, .avi, .mov, .mkv")
        print(f"[infer] 📁 File path: {video_path.absolute()}")
        raise
    
    if frames_data.shape[0] == 0:
        raise ValueError(f"Video file contains no frames: {video_path}")
    
    print(f"[infer] 📊 Video info: {frames_data.shape[0]} frames, {frames_data.shape[1:3]} resolution")
    
    # Use the same logic as dataset.py gen_vid()
    T = frames_data.shape[0]
    idxs = torch.linspace(0, T - 1, FIXED_NUM_FRAMES, dtype=torch.int)
    frames_data = frames_data[idxs]

    # Resize spatially and scale to [0, 1]
    frames_data = (
        torch.nn.functional.interpolate(
            frames_data.permute(0, 3, 1, 2).float(),       # (T, C, H, W)
            size=TARGET_SIZE,
            mode="bilinear",
            align_corners=False,
        ) / 255.0
    )

    # Reorder to (C, T, H, W) and add batch dimension
    frames_data = frames_data.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
    print(f"[infer] ✅ Video processed: {frames_data.shape} tensor")
    return frames_data.to(device, non_blocking=True)


def build_model(
    ldm_ckpt: str | pathlib.Path,
    cavp_ckpt: str | pathlib.Path,
    cfg: "OmegaConf | dict",
    device: torch.device | str,
) -> Tuple[LatentDiffusion, DPMSolverSampler]:
    """Create EnCodec, CAVP, UNet stack and load weights using existing build_unet."""

    device = torch.device(device)

    codec = EncodecWrapper(device=device).eval()
    cavp = CAVP_VideoOnly(str(cavp_ckpt)).to(device).eval()

    unet = build_unet(
        in_channels=cfg.model.latent_channels,
        model_channels=cfg.model.base_width,
        latent_w=cfg.data.latent_width,
        cross_attn_dim=cfg.model.cross_attention_dim,
    ).to(device)

    ldm = LatentDiffusion(
        codec=codec,
        unet=unet,
        cavp=cavp,
        timesteps=cfg.diffusion.timesteps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        guidance_prob=0.0,
        device=device,
    ).to(device)

    ckpt = torch.load(ldm_ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = ldm.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        print(f"[infer] ⚠️  state‑dict loaded with missing={len(missing)} unexpected={len(unexpected)}", file=sys.stderr)
        if len(missing) > 100:  # Too many missing keys suggests incompatible checkpoint
            print(f"[infer] ⚠️  Warning: Many missing keys ({len(missing)}). Check if checkpoint is compatible.", file=sys.stderr)

    sampler = DPMSolverSampler(ldm)
    ldm.eval()

    def fixed_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Fixed decode method that preserves channel dimension."""
        decoded = self.vae.decode(latents).sample[:, 0:1]  # Keep channel dimension
        spec = torch.nn.functional.interpolate(decoded, size=(self.mel_bins, self.T), mode="bilinear")
        return spec

    # Monkey-patch the decode method
    ldm.first_stage.decode = lambda z: fixed_decode(ldm.first_stage, z)

    return ldm, sampler


def mel_to_waveform(mel_db: torch.Tensor) -> torch.Tensor:
    """Quick & dirty Griffin‑Lim: (1,1,128,T) log‑mel → mono waveform."""
    print(f"[debug] Input mel_db stats - min: {mel_db.min():.3f}, max: {mel_db.max():.3f}, mean: {mel_db.mean():.3f}")
    
    # Clamp mel values to reasonable range
    mel_db = torch.clamp(mel_db, min=-80, max=20)  # Typical mel range
    
    mel_lin = 10.0 ** (mel_db.squeeze(0) / 10.0)
    print(f"[debug] After conversion mel_lin stats - min: {mel_lin.min():.3f}, max: {mel_lin.max():.3f}")

    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=N_FFT // 2 + 1,
        n_mels=128,
        sample_rate=SAMPLE_RATE,
        f_min=F_MIN,
        f_max=F_MAX,
    ).to(mel_lin.device)
    spec = inv_mel(mel_lin)
    print(f"[debug] Spectrogram stats - min: {spec.min():.3f}, max: {spec.max():.3f}")

    wave = torchaudio.functional.griffinlim(
        spec,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=torch.hann_window(N_FFT).to(mel_lin.device),
        power=2.0,
        n_iter=32,
        momentum=0.99,
        length=None,
        rand_init=True,
    )
    
    # Normalize the output waveform
    wave = torch.clamp(wave, min=-1.0, max=1.0)
    if wave.abs().max() > 0:
        wave = wave / wave.abs().max() * 0.8  # Scale to 80% to avoid clipping
    
    print(f"[debug] Final wave stats - min: {wave.min():.3f}, max: {wave.max():.3f}")
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
    p.add_argument("--config", type=str, default=None, help="YAML config file (default: configs/infer.yaml relative to project root)")
    p.add_argument("--seconds", type=float, default=4.0, help="length of audio to generate (sec)")
    p.add_argument("--steps", type=int, default=None, help="diffusion steps (None → take from config)")
    p.add_argument("--guidance", type=float, default=None, help="CFG scale (None → take from config)")

    # misc
    p.add_argument("--device", type=str, default="cuda", help="cuda | cpu | cuda:1 …")
    p.add_argument("--mux", action="store_true", help="mux generated audio back with original video")
    return p.parse_args()


def main():
    args = parse_args()

    # Validate required file paths
    print(f"[infer] 🔍 Validating input files...")
    
    # Check video file
    video_path = pathlib.Path(args.video)
    if not video_path.exists():
        print(f"[infer] ❌ Video file not found: {video_path.absolute()}")
        print(f"[infer] 💡 Please check the --video argument")
        sys.exit(1)
    
    # Check model checkpoints
    ldm_path = pathlib.Path(args.ldm_ckpt)
    if not ldm_path.exists():
        print(f"[infer] ❌ LDM checkpoint not found: {ldm_path.absolute()}")
        print(f"[infer] 💡 Please check the --ldm_ckpt argument")
        sys.exit(1)
        
    cavp_path = pathlib.Path(args.cavp_ckpt)
    if not cavp_path.exists():
        print(f"[infer] ❌ CAVP checkpoint not found: {cavp_path.absolute()}")
        print(f"[infer] 💡 Please check the --cavp_ckpt argument")
        sys.exit(1)

    # Handle config path resolution
    if args.config is None:
        config_path = get_project_root() / "configs" / "infer.yaml"
    else:
        config_path = pathlib.Path(args.config)
        if not config_path.is_absolute():
            # If relative path, make it relative to project root
            config_path = get_project_root() / config_path

    if not config_path.exists():
        print(f"[infer] ❌ Config file not found: {config_path}")
        print(f"[infer] Project root: {get_project_root()}")
        print(f"[infer] Current working directory: {os.getcwd()}")
        sys.exit(1)

    print(f"[infer] ✅ All files found")
    
    cfg = OmegaConf.load(config_path)
    steps = int(args.steps if args.steps is not None else cfg.inference.steps)
    guidance = float(args.guidance if args.guidance is not None else cfg.inference.cfg_scale)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[infer] 🖥️  Using device: {device}")

    # ---------------------------------------------------------------------
    print("[infer] building model …")
    ldm, sampler = build_model(args.ldm_ckpt, args.cavp_ckpt, cfg, device)
    
    # Fix the decode method to handle channel dimension properly
    def fixed_decode(first_stage_model, latents: torch.Tensor) -> torch.Tensor:
        """Fixed decode method that preserves channel dimension."""
        decoded = first_stage_model.vae.decode(latents).sample[:, 0:1]  # Keep channel dimension: (B, 1, H, W)
        spec = torch.nn.functional.interpolate(decoded, size=(first_stage_model.mel_bins, first_stage_model.T), mode="bilinear")
        return spec
    
    # Apply the fix
    ldm.first_stage.decode = lambda z: fixed_decode(ldm.first_stage, z)
    print("[infer] 🔧 Applied decode fix for channel dimension")

    # ---------------------------------------------------------------------
    print("[infer] encoding video …")
    frames = process_video_for_inference(args.video, device=device)
    
    # ---------------------------------------------------------------------
    print(f"[infer] sampling {args.seconds}s / {steps} steps  (CFG={guidance}) …")
    with torch.no_grad():
        # Calculate actual video duration from processed frames
        video_duration = FIXED_NUM_FRAMES / TARGET_FPS  # 40 frames / 25 fps = 1.6 seconds
        audio_time_steps = int(video_duration * SAMPLE_RATE // HOP_LENGTH)

        dummy_audio = torch.zeros(1, 1, 128, audio_time_steps, device=device)
        print(f"[infer] 🎵 Created dummy audio tensor: {dummy_audio.shape}")
        
        # Get video conditioning from CAVP encoder
        print(f"[infer] 🎬 Extracting video conditioning from CAVP encoder...")
        video_cond, _ = ldm.cond_stage(frames, dummy_audio)
        print(f"[infer] ✅ Video conditioning shape: {video_cond.shape}")
        
        video_cond = ldm.pe(video_cond)
        print(f"[infer] ✅ After positional encoding: {video_cond.shape}")
        
        # Use UNet directly instead of sampler
        unet = ldm.unet
        scheduler = ldm.sampler.scheduler
        print(f"[infer] 🔧 Using scheduler: {type(scheduler).__name__}")
        
        # Create random noise latent
        zT = torch.randn(1, ldm.latent_channels, ldm.latent_width, ldm.latent_width, device=device)
        print(f"[infer] 🎲 Initial noise tensor: {zT.shape}")
        
        # Set up scheduler
        scheduler.set_timesteps(steps, device=device)
        print(f"[infer] ⏱️  Scheduler timesteps: {len(scheduler.timesteps)} steps")
        x = zT
        
        # Run diffusion steps
        print(f"[infer] 🔄 Starting diffusion loop...")
        for i, t in enumerate(scheduler.timesteps):
            # Print progress every 10 steps or on first/last steps
            if i % 10 == 0 or i == len(scheduler.timesteps) - 1:
                print(f"[infer] Step {i+1}/{len(scheduler.timesteps)} (t={t})")
            
            # Ensure t is a tensor, not a tuple
            if isinstance(t, tuple):
                t = t[0] if len(t) > 0 else torch.tensor(0, device=device)
                print(f"[infer] ⚠️  Converted timestep tuple to tensor: {t}")
            
            # Ensure x is a tensor, not a tuple (from previous scheduler.step)
            if isinstance(x, tuple):
                print(f"[infer] ⚠️  x is tuple, extracting tensor: {type(x)} -> tensor")
                x = x[0]
            
            # Ensure x has the right shape for UNet
            if x.dim() == 4:  # (B, C, H, W)
                x_unet = x
            else:
                # Reshape if needed
                print(f"[infer] 🔄 Reshaping x from {x.shape} to UNet format")
                x_unet = x.view(1, ldm.latent_channels, ldm.latent_width, ldm.latent_width)
            
            # Check tensor shapes before UNet call
            if i == 0:  # Only print on first iteration to avoid spam
                print(f"[infer] 📊 UNet inputs - x: {x_unet.shape}, t: {t}, video_cond: {video_cond.shape}")
            
            if guidance == 1.0:
                eps = unet(x_unet, t, video_cond).sample
            else:
                eps_cond = unet(x_unet, t, video_cond).sample
                eps_uncond = unet(x_unet, t, torch.zeros_like(video_cond)).sample
                eps = eps_uncond + guidance * (eps_cond - eps_uncond)
            
            # Check epsilon shape
            if i == 0:
                print(f"[infer] 📊 UNet output eps: {eps.shape}")
            
            x = scheduler.step(eps, t, x_unet, return_dict=False)
            
            # Fix: Extract the tensor from the scheduler output
            if isinstance(x, tuple):
                if i == 0:  # Only print on first occurrence
                    print(f"[infer] 🔧 Scheduler returned tuple, extracting tensor")
                x = x[0]  # Take the first element (prev_sample)
            
            # Check final x shape
            if i == 0:
                print(f"[infer] 📊 Updated x shape: {x.shape}")
        
        print(f"[infer] ✅ Diffusion sampling complete!")
        z0 = x
        print(f"[infer] 🎯 Final latent z0 shape: {z0.shape}")
        
        print(f"[infer] 🔄 Decoding latent to mel-spectrogram...")
        mel_db = ldm.decode(z0)
        print(f"[infer] ✅ Decoded mel-spectrogram shape: {mel_db.shape}")

    print("[infer] mel → waveform …")
    waveform = mel_to_waveform(mel_db)
    print(f"[infer] 🎵 Generated waveform shape: {waveform.shape}")
    print(f"[infer] 💾 Saving audio to: {args.out_audio}")
    torchaudio.save(args.out_audio, waveform, SAMPLE_RATE)
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

    # Load and inspect the generated audio
    waveform, sample_rate = torchaudio.load(args.out_audio)
    print(f"Waveform shape: {waveform.shape}")
    print(f"Sample rate: {sample_rate}")
    print(f"Min value: {waveform.min():.6f}")
    print(f"Max value: {waveform.max():.6f}")
    print(f"RMS: {torch.sqrt(torch.mean(waveform**2)):.6f}")
    print(f"Non-zero samples: {torch.count_nonzero(waveform)}")


if __name__ == "__main__":
    main()
