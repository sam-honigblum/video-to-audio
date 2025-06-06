# =============================================================================
# Pre-trained EnCodec wrapper – replaces the hand-made VAE in Diff-Foley
# =============================================================================

from typing import List
import torch
import torch.nn as nn

class EncodecWrapper(nn.Module):
    """
    Wraps Meta's pretrained EnCodec neural codec so it looks like the
    AutoencoderKL planned for Diff-Foley.  The weights are loaded once and
    frozen; gradients flow only through the small projection layers.
    """

    def __init__(
        self,
        codebook_dim: int = 8,    # how many codebooks to concatenate
        target_sr: int = 24000,   # 16000, 24000, or 48000 supported
        device: str = "cuda",
    ):
        super().__init__()
        # ---- 1. load and freeze the codec -----------------------------------
        from audiocraft.models import CompressionModel
        ckpt_name = f"hance-ai/descript-audio-codec-{target_sr // 1000}khz"
        self.codec = CompressionModel.get_pretrained(ckpt_name).to(device)
        self.codec.eval().requires_grad_(False)

        # ---- 2. tiny trainable projection  ----------------------------------
        #    EnCodec returns a list of `codebook_dim` discrete codes per frame.
        #    We embed them and stack into a latent tensor (B, C=codebook_dim, T')
        # The codebook size is 1024 for the Descript model
        vocab_size = 1024
        self.code_embed = nn.Embedding(vocab_size, codebook_dim)

    # -------------------------------------------------------------------------
    # Public API identical to your planned AutoencoderKL
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        wav : (B, T) or (B, 1, T) waveform in [-1, 1] at `target_sr`
        Returns
        -------
        latents : (B, codebook_dim, T') float tensor for the UNet
        """
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)                # (B, 1, T)
        codes = self.codec.encode(wav)            # (B, n_q, T')
        # We only keep the first quantizer to match `codebook_dim`
        latents = self.code_embed(codes[:, 0])    # (B, T', codebook_dim)
        latents = latents.permute(0, 2, 1)        # (B, C, T')
        return latents.contiguous()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Inverse of `encode`.  We pick the nearest code in the embedding space
        and feed it to EnCodec's decoder.
        """
        # (B, C, T') → (B, T', C)
        latents = latents.permute(0, 2, 1)
        # cosine distance to every codebook vector
        codes = torch.cdist(
            latents.float(), self.code_embed.weight.float()[None, None]
        ).argmin(-1)                               # (B, T')
        # Create codes tensor with shape (B, n_q, T') where n_q is number of quantizers
        codes = torch.stack([codes] + [torch.zeros_like(codes) for _ in range(3)], dim=1)
        wav = self.codec.decode(codes)             # (B, 1, T)
        return wav.squeeze(1)                      # (B, T)
