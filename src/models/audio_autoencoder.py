# =============================================================================
# Pre-trained EnCodec wrapper – replaces the hand-made VAE in Diff-Foley
# =============================================================================

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
class EncodecWrapper(nn.Module):
    """
    Wraps Meta's pretrained EnCodec neural codec so it looks like the
    AutoencoderKL planned for Diff-Foley.  The weights are loaded once and
    frozen; gradients flow only through the small projection layers.

    We are now using stable diffusion v1.4 autoencoderkl vae
    repeat along channel dimension to simeulate img dimension
    and then avg along that dimension when dedcoding
    -Arnold
    """

    def __init__(
        self,
        mel_bins: int = 128,
        T: int = 641,
        latent_width: int = 64,
        latent_dim: int = 4,
        device: str = "cuda",
    ):
        super().__init__()
        # ---- 1. load and freeze the codec -----------------------------------
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
        self.vae.eval().requires_grad_(False)

        self.mel_bins = mel_bins
        self.T = T
        self.latent_width = latent_width
        self.latent_dim = latent_dim

    # -------------------------------------------------------------------------
    # Public API identical to your planned AutoencoderKL
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, spec) -> torch.Tensor:
        """
        args:
            spec (B, 1, mel_bins, T)
        return:
            latents (B, 4, latent_dim, latent_dim)
        """
        self.mel_bins = spec.shape[-2]
        self.T = spec.shape[-1]

        spec_resized = F.interpolate(spec, size=(512, 512), mode="bilinear") #resize into 256x256 for vae encoder input

        spec_resized = spec_resized.repeat(1, 3, 1, 1)

        encoded_dist = self.vae.encode(spec_resized)
        latents = encoded_dist.latent_dist.sample()
        print(latents.shape)
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents (B, 4, latent_dim, latent_dim)
        returns:
            spec (B, 1, mel_bins, T) ~ these are log mel spectrograms -> need to de-log and convert into wav afterwards
        """
        decoded = self.vae.decode(latents).sample[:, 0] #use only first channel

        spec = F.interpolate(decoded, size=(self.mel_bins, self.T), mode="bilinear")

        return spec
