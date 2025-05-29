# =============================================================================
# File: src/infer.py
# Role: Generate Foley audio for a silent video (Diff‑Foley reproduction)
# -----------------------------------------------------------------------------
# 1. Parse CLI arguments: silent MP4 path, YAML config, checkpoint, output name.
# 2. Build frozen EnCodecWrapper + frozen CAVP encoder + UNet with matching
#    in_channels.  Restore trained LatentDiffusion weights from checkpoint.
# 3. Read video frames → RGB tensor (B=1, 3, F, H, W) and normalise exactly
#    as during training (0‒1 then −0.5…0.5).
# 4. Call latent_diffusion.generate(video, seconds, steps, cfg_scale).
# 5. Save the resulting WAV and, if requested, mux it back into the MP4.
# -----------------------------------------------------------------------------
# Usage example:
#   python -m src.infer \
#       --video   assets/samples/silent_cat.mp4 \
#       --config  configs/infer.yaml           \
#       --ckpt    checkpoints/best.pt          \
#       --out     outputs/cat_foley.wav        \
#       --mux                      # (optional)
# =============================================================================

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch
import torchaudio
import torchvision.io as io
from omegaconf import OmegaConf

from models.audio_autoencoder import EncodecWrapper
from models.video_encoder import VideoEncoder, load_pretrained_cavp
from models.unet import build_unet
from models.latent_diffusion import LatentDiffusion
from models.sampler import DPMSolverSampler
from models.unet import build_unet


# -----------------------------------------------------------------------------
# helper — video to tensor exactly like the training loader
# -----------------------------------------------------------------------------

def load_rgb_video(path: Path, target_fps: int = 25) -> torch.Tensor:
    """Return (1, 3, F, H, W) float32 in −0.5…0.5 range."""
    frames, _, info = io.read_video(str(path), pts_unit="sec")     # (F, H, W, 3), uint8
    # Frame rate normalisation (simple nearest‑frame drop / repeat)
    if int(info["video_fps"]) != target_fps:
        idx = torch.linspace(0, len(frames) - 1, int(len(frames) * target_fps / info["video_fps"]))
        frames = frames[idx.long()]
    frames = frames.permute(0, 3, 1, 2).float() / 255.0             # (F, 3, H, W) 0‑1
    frames = frames - 0.5                                           # centre
    return frames.unsqueeze(0)                                      # (1, 3, F, H, W)

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Diff‑Foley inference")
    parser.add_argument("--video", required=True, type=Path, help="Path to silent MP4")
    parser.add_argument("--config", default=Path("configs/infer.yaml"), type=Path)
    parser.add_argument("--ckpt",   required=True, type=Path, help="Trained stage‑2 checkpoint .pt")
    parser.add_argument("--out",    default=None, type=Path, help="Output WAV file")
    parser.add_argument("--mux",    action="store_true", help="Mux WAV back into MP4")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ build
    codec = EncodecWrapper(target_sr=cfg.sample_rate).to(device).eval()
    cavp  = load_pretrained_cavp().to(device).eval()                # frozen

    in_ch = codec.code_embed.embedding_dim                         # 8 for 24 kHz
    unet  = build_unet(in_channels=in_ch, base_width=cfg.model.base_width).to(device)

    ld_model = LatentDiffusion(
        unet=unet,
        timesteps=cfg.steps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        guidance_prob=cfg.guidance.classifier_free,
        target_sr=cfg.sample_rate,
        device=device,
    ).to(device)

    ld_state = torch.load(args.ckpt, map_location="cpu")
    ld_model.load_state_dict(ld_state["model"], strict=False)
    ld_model.eval()

    # ------------------------------------------------------------------ video → tensor
    video_tensor = load_rgb_video(args.video, target_fps=cfg.data.fps).to(device)
    seconds = len(video_tensor[0, 0]) / cfg.data.fps

    # ------------------------------------------------------------------ generate
    sampler_cls = DPMSolverSampler #if cfg.sampler == "dpm_solver" else DDIMSampler
    ld_model.sampler = sampler_cls(ld_model)

    wav = ld_model.generate(
        video_tensor, seconds=seconds, steps=cfg.steps, guidance=cfg.cfg_scale
    )

    # ------------------------------------------------------------------ save
    out_path = args.out or args.video.with_suffix("_foley.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), wav.unsqueeze(0).cpu(), sample_rate=codec.codec.sample_rate)
    print(f"[✓] Saved audio to {out_path.relative_to(Path.cwd())}")

    if args.mux:
        muxed_path = args.video.with_suffix("_foley.mp4")
        cmd = [
            "ffmpeg", "-y", "-i", str(args.video), "-i", str(out_path),
            "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", str(muxed_path)
        ]
        subprocess.run(cmd, check=True)
        print(f"[✓] Muxed audio into {muxed_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
