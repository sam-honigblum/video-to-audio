# =============================================================================
# File: src/models/latent_diffusion.py
# Role: UNet-based latent diffusion core with video cross-attention
# -----------------------------------------------------------------------------
# • UNet backbone: attention at res 4, 2, 1.
# • Cross-Attention injects video tokens after every block.
# • register_schedule() pre-computes β_t, ᾱ_t tables.
# • p_losses(), p_sample_loop() implement DDPM training + double guidance hook.
# =============================================================================
# =============================================================================
# File: src/models/latent_diffusion.py
# Role: UNet-based latent diffusion core with video cross-attention
# -----------------------------------------------------------------------------
# • Uses a *frozen* EnCodec audio codec (EncodecWrapper) – no Stage-1 training.
# • Treats the 1-D EnCodec latent sequence as a 4-D map (B, C, 1, W)
#   so the existing UNet interface remains unchanged.
# • register_schedule() pre-computes β-tables for DDPM/DDIM.
# • p_losses(), generate() support classifier-free guidance.
# =============================================================================

import torch
import torch.nn as nn
from torch.optim import AdamW

from .audio_autoencoder import EncodecWrapper          # NEW
from .video_encoder     import VideoEncoder
from .sampler           import DDIMSampler

# -----------------------------------------------------------------------------


class LatentDiffusion(nn.Module):
    """
    Latent-diffusion model adapted to EnCodec latents.
    """

    def __init__(
        self,
        unet: nn.Module,
        timesteps: int = 1_000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        guidance_prob: float = 0.2,
        latent_width: int = 160,  # audio frames after compression (≈ video_len · 4)
        target_sr: int = 24_000,  # keep in sync with EnCodec checkpoint
        device: str = "cuda",
    ):
        super().__init__()

        # ------------------------------------------------------------------ 1
        # Pre-trained audio codec (frozen)  +  frozen video encoder
        # ----------------------------------------------------------------------
        self.first_stage: EncodecWrapper = (
            EncodecWrapper(target_sr=target_sr).to(device).eval()
        )
        self.cond_stage: VideoEncoder = VideoEncoder().to(device).eval()
        for p in self.first_stage.parameters():
            p.requires_grad = False
        for p in self.cond_stage.parameters():
            p.requires_grad = False

        # Trainable UNet
        self.unet = unet

        # DDIM / DPM sampler wrapper
        self.sampler = DDIMSampler(self)

        # ------------------------------------------------------------------ 2
        # Pre-compute β / ᾱ lookup tables
        # ----------------------------------------------------------------------
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)

        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt()
        )
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())

        # ------------------------------------------------------------------ 3
        self.guidance_prob = guidance_prob
        self.latent_channels = self.first_stage.code_embed.embedding_dim  # 8
        self.latent_width = latent_width  # default 160 columns

    # ======================================================================
    #                           helper methods
    # ======================================================================

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav : (B, T) or (B, 1, T) waveform in the target_sr domain.
        returns : (B, C, 1, W) latent tensor
        """
        z = self.first_stage.encode(wav)          # (B, C, W)
        return z.unsqueeze(2)                     # add dummy height: (B, C, 1, W)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (B, C, 1, W) latent tensor
        returns : (B, T) waveform
        """
        z = z.squeeze(2)                          # (B, C, W)
        return self.first_stage.decode(z)

    # ------------------------------------------------------------------
    def q_sample(self, z0, t, noise=None):
        """
        Forward (noising) process q(z_t | z_0).
        """
        if noise is None:
            noise = torch.randn_like(z0)
        sqrt_ᾱ = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1m = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ᾱ * z0 + sqrt_1m * noise, noise

    # ======================================================================
    #                         training forward pass
    # ======================================================================

    def forward(self, wav, video):
        """
        wav   : raw waveform (B, T)
        video : RGB tensor (B, C, F, H, W)
        """
        z0 = self.encode(wav)  # (B, C, 1, W)
        t = torch.randint(0, len(self.alphas_cumprod), (z0.size(0),), device=z0.device)
        xt, eps = self.q_sample(z0, t)

        cond = self.cond_stage(video)             # (B, 40, 512)
        if torch.rand(()) < self.guidance_prob:
            cond = torch.zeros_like(cond)         # classifier-free dropout

        eps_hat = self.unet(xt, t, cond)          # predict the noise
        return nn.functional.mse_loss(eps_hat, eps)

    # ======================================================================
    #                            inference
    # ======================================================================

    @torch.no_grad()
    def generate(self, video, seconds: float, steps: int = 50, guidance: float = 2.0):
        """
        video   : RGB tensor (B, C, F, H, W)
        seconds : desired audio duration
        """
        cond = self.cond_stage(video)

        # Estimate latent width W from waveform length: W ≈ sr / 320 * seconds
        W = int(seconds * (self.first_stage.codec.sample_rate / 320))
        zT = torch.randn(
            video.size(0), self.latent_channels, 1, W, device=video.device
        )

        z0 = self.sampler.ddim_sample(zT, cond, steps, guidance)
        return self.decode(z0)

    # ======================================================================
    #                       optimizer convenience
    # ======================================================================

    def configure_optim(self, lr: float = 1e-4):
        return AdamW(self.unet.parameters(), lr=lr)
