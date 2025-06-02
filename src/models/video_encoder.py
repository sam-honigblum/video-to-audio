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
        self.video_temporal_pool = nn.MaxPool1d(kernel_size=16)

        # ---------------- audio branch ----------------
        self.audio_encoder = Cnn14(feat_dim=feat_dim)
        self.audio_temporal_pool = nn.MaxPool1d(kernel_size=16)

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
        # -------- video --------
        v = self.video_encoder(video)            # (B, 2048, T, H', W')
        b, c, t, h, w = v.shape
        v = v.view(b * t, c, h, w)
        v = F.avg_pool2d(v, kernel_size=h)       # (B·T, 2048, 1, 1)
        v = v.view(b, t, c)                      # (B, T, 2048)
        v = self.video_projection(v)             # (B, T, D)
        v = self.video_temporal_pool(v.permute(0, 2, 1)).squeeze(-1)  # (B, D)
        v = F.normalize(v, dim=-1)

        # -------- audio --------
        s = spectrogram.permute(0, 1, 3, 2)      # → (B, 1, T, mel_bins)
        s = self.audio_encoder(s)                # (B, T, D)
        s = self.audio_temporal_pool(s.permute(0, 2, 1)).squeeze(-1)  # (B, D)
        s = F.normalize(s, dim=-1)

        return v, s

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

# -----------------------------------------------------------------------------
class CAVP_Loss(nn.Module):
    """Combined semantic + temporal loss from Diff‑Foley.

    L_total = L_semantic + λ · L_temporal
    """

    def __init__(self, shared_logit_scale: nn.Parameter, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_
        self.clip_loss = _CLIPStyleLoss(shared_logit_scale)

    # ------------------------------------------------------------------
    def forward(self, audio_feats: torch.Tensor, video_feats: torch.Tensor, vid_ids: torch.Tensor) -> torch.Tensor:
        """Return total loss.

        Parameters
        ----------
        audio_feats, video_feats : (B, D)
        vid_ids                 : (B,) – integer source‑video identifiers
        """
        semantic = self.semantic_loss(audio_feats, video_feats)
        temporal = self.temporal_loss(audio_feats, video_feats, vid_ids)
        return semantic + self.lambda_ * temporal

    # ------------------------------------------------------------------
    @staticmethod
    def _make_temporal_labels(vid_ids: torch.Tensor) -> torch.Tensor:
        """Assign each sample a positive index that is *another* clip from the same video.

        If a video appears only once in the batch, fall back to the diagonal.
        """
        device = vid_ids.device
        B = len(vid_ids)
        labels = torch.arange(B, device=device)
        # build an index list for each unique video id
        unique_ids = vid_ids.unique()
        for uid in unique_ids:
            idx = (vid_ids == uid).nonzero(as_tuple=False).flatten()
            if idx.numel() > 1:
                rolled = torch.roll(idx, shifts=-1)
                labels[idx] = rolled
        return labels

    # ------------------------------------------------------------------
    def semantic_loss(self, audio_feats: torch.Tensor, video_feats: torch.Tensor) -> torch.Tensor:
        """Diagonal aligns matching audio–video clips (standard CLIP)."""
        labels = torch.arange(audio_feats.size(0), device=audio_feats.device)
        return self.clip_loss(audio_feats, video_feats, labels)

    # ------------------------------------------------------------------
    def temporal_loss(self, audio_feats: torch.Tensor, video_feats: torch.Tensor, vid_ids: torch.Tensor) -> torch.Tensor:
        """Align *different* timestamps from the same source video."""
        labels = self._make_temporal_labels(vid_ids)
        return self.clip_loss(audio_feats, video_feats, labels)
