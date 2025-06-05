# =============================================================================
# File: src/models/video_encoder.py (patched)
# Role: Frozen CAVP backbone that maps RGB video to 40×512 tokens
#         + Corrected CAVP loss implementation (semantic + temporal)
# =============================================================================

import torch
from torch import nn
import torch.nn.functional as F
import math

from .cavp_modules import Cnn14, ResNet3dSlowOnly

# -----------------------------------------------------------------------------
# Encoders
# -----------------------------------------------------------------------------
class CAVP(nn.Module):
    """Video ⇄ Audio encoders with fixed projections used during diffusion."""

    def __init__(self, feat_dim: int = 512, temperature: float = 0.07):
        super().__init__()

        # ---------------- video branch ----------------
        self.video_encoder = ResNet3dSlowOnly(depth=50, pretrained=None)
        self.video_projection = nn.Linear(2048, feat_dim)
        self.video_max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.video_mean_pool = nn.AdaptiveAvgPool1d(output_size=1)

        # ---------------- audio branch ----------------
        self.audio_encoder = Cnn14(feat_dim=feat_dim)
        self.audio_max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.audio_mean_pool = nn.AdaptiveAvgPool1d(output_size=1)

        # shared learnable temperature (τ⁻¹)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

    # ---------------------------------------------------------------------
    def forward(self, video: torch.Tensor, spectrogram: torch.Tensor):
        """Returns l2‑normalised features ready for the CAVP loss.

        Parameters
        ----------
        video : (B, C, T, H, W)
        spectrogram : (B, 1, mel_bins, T)
        """


# -----------------------------------------------------------------------------
# Contrastive losses
# -----------------------------------------------------------------------------
class _CLIPStyleLoss(nn.Module):
    """OpenCLIP‑style InfoNCE with a *single* positive index per row.

    Parameters
    ----------
    shared_logit_scale : nn.Parameter
        The temperature parameter (log space) shared across the whole model.
    """

    def __init__(self, shared_logit_scale: nn.Parameter):
        super().__init__()
        self.logit_scale = shared_logit_scale

    # ------------------------------------------------------------------
    def forward(self, audio: torch.Tensor, video: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute symmetric KL‑divergence as in CLIP.

        audio, video : (B, D) – l2‑normalised feature vectors
        labels       : (B,)   – index of the *positive* video for each audio row
        """
        a = F.normalize(audio, dim=-1)
        v = F.normalize(video, dim=-1)

        logits = self.logit_scale.exp().clamp(max=100) * (a @ v.T)  # (B, B)

        loss_a2v = F.cross_entropy(logits, labels)
        loss_v2a = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_a2v + loss_v2a)

        # video encode
        video_feat = self.video_encoder(video)
        b, c, t, h, w = video_feat.shape
        video_feat = F.avg_pool2d(video_feat.view(-1, h, w), kernel_size=h)
        video_feat = video_feat.reshape(b, c, t).permute(0, 2, 1)
        video_feat = self.video_projection(video_feat)
        video_max = self.video_max_pool(video_feat.permute(0, 2, 1)).squeeze(-1)
        video_mean = self.video_mean_pool(video_feat.permute(0, 2, 1)).squeeze(-1)
        video_max = F.normalize(video_max, dim=-1)
        video_mean = F.normalize(video_mean, dim=-1)

        # audio encode
        spectrogram = spectrogram.permute(0, 1, 3, 2) # (B, 1, T, mel_num)
        spectrogram_feat = self.audio_encoder(spectrogram) #(B, T, C)
        spectrogram_max = self.audio_max_pool(spectrogram_feat.permute(0, 2, 1)).squeeze(-1)
        spectrogram_mean = self.audio_mean_pool(spectrogram_feat.permute(0, 2, 1)).squeeze(-1)
        spectrogram_max = F.normalize(spectrogram_max, dim=-1)
        spectrogram_mean = F.normalize(spectrogram_mean, dim=-1)

        return video_max, video_mean, spectrogram_max, spectrogram_mean, self.logit_scale.exp()


# -----------------------------------------------------------------------------
class CAVP_Loss(nn.Module):
    """Combined semantic + temporal loss from Diff‑Foley.

    L_total = L_semantic + λ · L_temporal
    """
    Implements   L_total = L_extra + λ · L_intra
    where
        L_extra  - semantic contrast  (different videos)
        L_intra  - temporal contrast  (other segments of same video)
    """
    def __init__(self, clip_num: int = 2, lambda_: float = 1.0):
        super().__init__()
        self.lambda_   = lambda_
        self.clip_num = clip_num

    def forward(
        self,
        video_feats: torch.Tensor,       # (B, D)
        video_mean_feats: torch.Tensor,
        audio_feats: torch.Tensor,       # (B, D)
        audio_mean_feats: torch.Tensor,       # (B, D)
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:

        logits_per_vid = logit_scale * torch.matmul(video_feats, audio_feats.T)
        logits_per_aud = logit_scale * torch.matmul(audio_feats, video_feats.T)

        num_logits = logits_per_vid.shape[0]
        labels = torch.arange(num_logits, device=video_feats.device, dtype=torch.long)
        extra_loss = (F.cross_entropy(logits_per_vid, labels) + F.cross_entropy(logits_per_aud, labels)) / 2.0

        batches, dim = video_mean_feats.shape
        vid_intra_feats = video_mean_feats.reshape(-1, self.clip_num, dim)
        aud_intra_feats = audio_mean_feats.reshape(-1, self.clip_num, dim)

        intra_logits_per_vid = logit_scale * torch.matmul(vid_intra_feats, aud_intra_feats.permute(0, 2, 1))
        intra_logits_per_aud = logit_scale * torch.matmul(aud_intra_feats, vid_intra_feats.permute(0, 2, 1))

        intra_batches, intra_num_logits, _ = intra_logits_per_vid.shape
        intra_logits_per_vid = intra_logits_per_vid.reshape(intra_batches * intra_num_logits, intra_num_logits)
        intra_logits_per_aud = intra_logits_per_aud.reshape(intra_batches * intra_num_logits, intra_num_logits)

        intra_labels = torch.arange(intra_num_logits, device=video_mean_feats.device, dtype=torch.long).unsqueeze(0).repeat(intra_batches, 1).flatten() # create labels for everything all batches

        intra_loss = (F.cross_entropy(intra_logits_per_vid, intra_labels) + F.cross_entropy(intra_logits_per_aud, intra_labels)) / 2.0
        return extra_loss + self.lambda_ * intra_loss

