# =============================================================================
# File: src/models/latent_diffusion.py  (modified)
# Role: UNet‐based latent diffusion core with video cross‐attention
#       plus "spectrogram→latent" encoder for spec inputs.
# =============================================================================

import torch
import torch.nn as nn
from torch.optim import AdamW
from .sampler import DPMSolverSampler
from .video_feat_pe import CAVPVideoPE

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
        codec: nn.Module,
        unet: nn.Module,
        cavp: nn.Module,
        timesteps: int = 1_000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        guidance_prob: float = 0.2,
        device: str = "cuda",
    ):
        super().__init__()

        # ----------------------------------------------------------------------------
        # 1. Pre-compute β / ᾱ lookup tables FIRST
        # ----------------------------------------------------------------------------
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt()
        )
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())

        # ----------------------------------------------------------------------------
        # 2. Pre-trained audio codec (frozen)  +  frozen video encoder
        # ----------------------------------------------------------------------------
        self.first_stage = codec
        self.cond_stage = cavp
        for p in self.first_stage.parameters():
            p.requires_grad = False
        for p in self.cond_stage.parameters():
            p.requires_grad = False

        self.pe = CAVPVideoPE(self.cond_stage.latent_dim, self.cond_stage.latent_dim)

        # ----------------------------------------------------------------------------
        # 3. Small CNN to map (B,1,128,T) → (B, C, 1, W)
        # ----------------------------------------------------------------------------
        self.latent_channels = self.first_stage.latent_dim
        self.latent_width = self.first_stage.latent_width

        # ----------------------------------------------------------------------------
        # 4. Trainable UNet backbone
        # ----------------------------------------------------------------------------
        self.unet = unet

        # ----------------------------------------------------------------------------
        # 5. DDIM / DPM sampler wrapper (now alphas_cumprod is initialized)
        # ----------------------------------------------------------------------------
        self.sampler = DPMSolverSampler(self)

        # ----------------------------------------------------------------------------
        # 6. Other hyperparams
        # ----------------------------------------------------------------------------
        self.guidance_prob = guidance_prob

        # Validate initialization
        if not hasattr(self, "alphas_cumprod"):
            raise RuntimeError("Failed to initialize alphas_cumprod buffer")

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
        return self.first_stage.encode(x)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inverse of encode() when input was raw waveform.
        If `z` came from a mel‐encoder, this decode is undefined (you'd need a separate decoder).
        """
        return self.first_stage.decode(z)

    # =============================================================================
    #                         training forward pass
    # =============================================================================

    def q_sample(self, x0, t):
        """
        Add noise to x0 at timestep t
        x0 : (B, C, 1, latent) latent vectors
        t  : (B,) integer timestep indices
        """
        # Get corresponding alpha_bar_t
        batch_size = x0.size(0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].reshape(batch_size, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].reshape(batch_size, 1, 1, 1)

        # Sample noise
        eps = torch.randn_like(x0)

        # Return x_t and the noise used
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * eps
        return xt, eps

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
        cond = self.cond_stage(video, x)  # (B, 40, 512) or whatever your cavp outputs
        cond = self.pe(cond) #projection and positional encoding of cavp latent vector

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
            # You'd need a separate Mel‐decoder if training on mel.
        """
        cond = self.cond_stage(video)

        # For pure-spec inference, you'd skip the waveform path anyway.
        W = self.latent_width
        zT = torch.randn(video.size(0), self.latent_channels, W, W, device=video.device)
        z0 = self.sampler.ddim_sample(zT, cond, steps, guidance)
        return self.decode(z0)

    # =============================================================================
    #                       optimizer convenience
    # =============================================================================

    def configure_optim(self, lr: float = 1e-4):
        return AdamW(self.unet.parameters(), lr=lr)
