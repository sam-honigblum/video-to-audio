# =============================================================================
# File: src/models/latent_diffusion.py  (modified)
# Role: UNet‐based latent diffusion core with video cross‐attention
#       plus “spectrogram→latent” encoder for spec inputs.
# =============================================================================

import torch
import torch.nn as nn
from torch.optim import AdamW
from .audio_autoencoder import EncodecWrapper
from .cavp_encoder     import CAVP as VideoEncoder
from .sampler           import DPMSolverSampler

class LatentDiffusion(nn.Module):
    """
    Latent-diffusion model adapted to either:
      • a raw waveform (via EncodecWrapper)  →  latent (B,C,1,W)
      • OR a mel-spectrogram (B,1,128,T)      →  latent (B,C,1,W) via spec_encoder

    Inference still calls `encode(...)` internally, so as long as you pass
    either a waveform or a spectrogram, it will route correctly.
    """

    def __init__(
        self,
        unet: nn.Module,
        timesteps: int = 1_000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        guidance_prob: float = 0.2,
        latent_width: int = 32,     # final W
        target_sr: int = 24_000,
        device: str = "cuda",
    ):
        super().__init__()

        # ----------------------------------------------------------------------------
        # 1. Pre-trained audio codec (frozen)  +  frozen video encoder
        # ----------------------------------------------------------------------------
        self.first_stage: EncodecWrapper = (
            EncodecWrapper(target_sr=target_sr).to(device).eval()
        )
        self.cond_stage: VideoEncoder = VideoEncoder().to(device).eval()
        for p in self.first_stage.parameters():
            p.requires_grad = False
        for p in self.cond_stage.parameters():
            p.requires_grad = False

        # ----------------------------------------------------------------------------
        # 1.b  Small CNN to map (B,1,128,T) → (B, C, 1, W)
        #     where C = latent_channels (e.g. 8), W = latent_width (32)
        # ----------------------------------------------------------------------------
        self.latent_channels = self.first_stage.code_embed.embedding_dim  # e.g. 8
        self.latent_width    = latent_width

        # This spec_encoder collapses 128 mel‐bins into latent_channels,
        # then downsamples/pads time to exactly latent_width columns.
        self.spec_encoder = nn.Sequential(
            # Conv over mel‐bins: kernel (128,1) → "squeeze" mel dimension
            nn.Conv2d(
                in_channels=1,
                out_channels=self.latent_channels,
                kernel_size=(128, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
            nn.ReLU(inplace=True),
            # Now shape is (B, latent_channels, T), but in 2D conv format: (B, C, 1, T)
            # Next: pad or interpolate along the W axis to exactly latent_width
            nn.Upsample(size=(1, self.latent_width), mode="bilinear", align_corners=False),
        )

        # ----------------------------------------------------------------------------
        # 2. Trainable UNet backbone
        # ----------------------------------------------------------------------------
        self.unet = unet

        # ----------------------------------------------------------------------------
        # 3. DDIM / DPM sampler wrapper
        # ----------------------------------------------------------------------------
        self.sampler = DPMSolverSampler(self)

        # ----------------------------------------------------------------------------
        # 4. Pre-compute β / ᾱ lookup tables
        # ----------------------------------------------------------------------------
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())

        # ----------------------------------------------------------------------------
        # 5. Other hyperparams
        # ----------------------------------------------------------------------------
        self.guidance_prob = guidance_prob

    # =============================================================================
    #                           helper methods
    # =============================================================================

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        If `x` is a raw waveform:  x.shape = (B, T) or (B,1,T):
            → call EncodecWrapper.encode(wav) → (B, C, W_true) → squeeze to (B,C,1,W)

        If `x` is a spectrogram: x.shape = (B, 1, 128, T_frame):
            → pass through `spec_encoder` to get (B, C, 1, W=latent_width)

        Returns:
            latent z:  shape = (B, C, 1, W) exactly
        """
        # Case 1: Raw waveform
        if x.ndim == 2 or (x.ndim == 3 and x.shape[1] == 1):
            # let EnCodec produce its own latent_width “W_true”
            z = self.first_stage.encode(x)   # (B, C, W_true)
            W_true = z.shape[-1]
            # If W_true != latent_width, pad or truncate
            if W_true > self.latent_width:
                z = z[..., : self.latent_width]
            elif W_true < self.latent_width:
                pad = torch.zeros_like(z[..., :1]).repeat(1, 1, self.latent_width - W_true)
                z = torch.cat([z, pad], dim=-1)
            return z.unsqueeze(2)  # → (B, C, 1, latent_width)

        # Case 2: Mel-spectrogram input
        elif x.ndim == 4 and x.shape[1] == 1 and x.shape[2] == 128:
            # x: (B, 1, 128, T_frame)
            z = self.spec_encoder(x)  # (B, C, 1, latent_width)
            return z

        else:
            raise ValueError(
                f"Unsupported shape for encode(): {x.shape}. "
                "Expected waveform (B,T) or (B,1,T) or mel-spec (B,1,128,T_frame)."
            )

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inverse of encode() when input was raw waveform.
        If `z` came from a mel‐encoder, this decode is undefined (you’d need a separate decoder).
        """
        z = z.squeeze(2)  # (B, C, W)
        return self.first_stage.decode(z)

    # =============================================================================
    #                         training forward pass
    # =============================================================================

    def forward(self, x, video):
        """
        x     : either waveform (B,T) or spectrogram (B,1,128,T_frame)
        video : RGB tensor (B, 3, F, H, W)
        """
        # 1. Encode x into latent z0
        z0 = self.encode(x)  # → (B, C, 1, latent_width)

        # 2. Sample random t and produce noisy latent
        t = torch.randint(0, len(self.alphas_cumprod), (z0.size(0),), device=z0.device)
        xt, eps = self.q_sample(z0, t)

        # 3. Condition on video
        cond = self.cond_stage(video, x)     # (B, 40, 512) or whatever your VideoEncoder outputs
        if torch.rand(()) < self.guidance_prob:
            cond = torch.zeros_like(cond)

        # 4. Predict noise with UNet
        eps_hat = self.unet(xt, t, cond)
        return nn.functional.mse_loss(eps_hat, eps)

    # =============================================================================
    #                            inference
    # =============================================================================

    @torch.no_grad()
    def generate(self, video, seconds: float, steps: int = 50, guidance: float = 2.0):
        """
        video   : RGB tensor (B, 3, F, H, W)
        seconds : desired audio duration in seconds

        To generate from a spectrogram instead of waveform, call:
            zT = torch.randn(B, C, 1, latent_width)
            eps_latent = sampler.ddim_sample(zT, cond, steps, guidance)
            # Cannot decode an mel‐encoded latent via `decode()`. 
            # You’d need a separate Mel‐decoder if training on mel.
        """
        cond = self.cond_stage(video)

        # For pure-spec inference, you'd skip the waveform path anyway.
        W = self.latent_width
        zT = torch.randn(video.size(0), self.latent_channels, 1, W, device=video.device)
        z0 = self.sampler.ddim_sample(zT, cond, steps, guidance)
        return self.decode(z0)

    # =============================================================================
    #                       optimizer convenience
    # =============================================================================

    def configure_optim(self, lr: float = 1e-4):
        return AdamW(self.unet.parameters(), lr=lr)
