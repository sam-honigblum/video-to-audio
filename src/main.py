# =============================================================================
# File: src/infer.py
# Role: Generate Foley audio for a silent video (Diff‑Foley reproduction)
# -----------------------------------------------------------------------------
#   1. Parse CLI arguments.
#   2. Load YAML config, create EnCodecWrapper, pre‑trained CAVP, UNet, and
#      LatentDiffusion; restore the trained weights.
#   3. Convert the input MP4 into a tensor (B, 3, F, H, W) in the range [-1,1].
#   4. Call model.generate(video_tensor, seconds) to produce a 24‑kHz waveform.
#   5. Save WAV.  If --mux is passed, mux the WAV back into the MP4 with FFmpeg.
"""
python -m src.main infer \
       --video samples/silent.mp4 \
       --config configs/infer.yaml \
       --checkpoint checkpoints/best.pt \
       --out_wav outputs/foley.wav \
       --mux              # add this flag to embed audio back into MP4
"""

# =============================================================================

import argparse, subprocess, os, sys, math, time
from pathlib import Path

import torch, torchaudio
from torchvision.io import read_video
from omegaconf import OmegaConf

# --- project imports ---------------------------------------------------------
from src.models.audio_autoencoder import EncodecWrapper
from src.models.video_encoder     import VideoEncoder
from src.models.latent_diffusion  import LatentDiffusion
from src.models.unet              import build_unet            # you created this

# -----------------------------------------------------------------------------

def video_to_tensor(video_path: str, fps_target: int = 25):
    """Load MP4 → RGB float tensor (1, 3, F, H, W) in [-1, 1]."""
    frames, orig_audio, info = read_video(video_path, pts_unit="sec")  # (F, H, W, 3)
    # Downsample / upsample frames to fps_target by simple stride
    orig_fps = info["video_fps"]
    stride = max(1, round(orig_fps / fps_target))
    frames = frames[::stride]
    frames = frames.permute(0, 3, 1, 2).float() / 255.0              # (F, 3, H, W) 0‑1
    frames = frames * 2.0 - 1.0                                       # −1 … +1
    return frames.unsqueeze(0)                                        # (1, 3, F, H, W)

# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Diff‑Foley inference")
    p.add_argument("--video", required=True, help="path to silent MP4")
    p.add_argument("--config", default="configs/infer.yaml")
    p.add_argument("--checkpoint", required=True, help="trained LDM weights")
    p.add_argument("--out_wav", default="generated.wav")
    p.add_argument("--mux", action="store_true", help="mux WAV back into MP4")
    return p.parse_args()

# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---------------- config merge ---------------------------------------
    cfg_default = OmegaConf.load("configs/default.yaml")
    cfg_user    = OmegaConf.load(args.config)
    cfg         = OmegaConf.merge(cfg_default, cfg_user)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- build codec & video encoder ------------------------
    codec = EncodecWrapper(target_sr=24000).to(device).eval().requires_grad_(False)
    video_enc = VideoEncoder().to(device).eval().requires_grad_(False)

    # ---------------- build UNet & LDM -----------------------------------
    unet = build_unet(in_channels=codec.code_embed.embedding_dim).to(device)
    model = LatentDiffusion(unet=unet, target_sr=24000, device=device).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # ---------------- video tensor ---------------------------------------
    vid_tensor = video_to_tensor(args.video, fps_target=cfg.data.get("video_fps", 25)).to(device)
    seconds = (vid_tensor.shape[2] / cfg.data.get("video_fps",25))

    # ---------------- generate -------------------------------------------
    with torch.no_grad():
        wav = model.generate(
            vid_tensor,
            seconds=seconds,
            steps=cfg.infer.get("steps", 50),
            guidance=cfg.infer.get("cfg_scale", 3.0),
        )  # (B, T)

    wav = wav.cpu().unsqueeze(0)  # (1, 1, T) torchaudio expects (channels, T) or (batch, channels, T)
    torchaudio.save(args.out_wav, wav, sample_rate=24000)
    print(f"[✓] Saved {args.out_wav}")

    # ---------------- optional mux ---------------------------------------
    if args.mux:
        mux_out = Path(args.video).with_suffix("_foley.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", args.video,
            "-i", args.out_wav,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(mux_out),
        ]
        subprocess.run(cmd, check=True)
        print(f"[✓] Muxed into {mux_out}")


if __name__ == "__main__":
    main()
