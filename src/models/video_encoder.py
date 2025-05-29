# =============================================================================
# File: src/models/video_encoder.py
# Role: Frozen CAVP backbone that maps RGB video to 40×512 tokens
# -----------------------------------------------------------------------------
# • Wraps a pretrained 3D-ViT (TimeSformer) fine-tuned with CAVP loss.
# • Adds positional encodings so cross-attention lines up temporally.
# • forward(video) → (B, 40, 512) for UNet cross-attention.
# =============================================================================

import torch
from torch import nn
import torch.nn.functional as F
import math

from cavp_modules import Cnn14, ResNet3dSlowOnly

class CAVP(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()

        self.video_encoder = ResNet3dSlowOnly(depth=50)
        self.audio_encoder = Cnn14()

    def forward(self, x):

        pass

class CAVP_Loss(nn.Module):
    """
    Implements   L_total = L_extra + λ · L_intra
    where
        L_extra  - semantic contrast  (different videos)
        L_intra  - temporal contrast  (other segments of same video)
    """
    def __init__(self, lambda_: float = 1.0, temperature: float = 0.07):
        super().__init__()
        self.lambda_   = lambda_
        self.clip_loss = _CLIPStyleLoss(logit_scale_init=math.log(1/temperature))

    # ======= public entry =================================================
    def forward(
        self,
        audio_feats: torch.Tensor,       # (B, D)
        video_feats: torch.Tensor,       # (B, D)
        vid_ids: torch.Tensor            # (B,)  int identifier per *source video*
    ) -> torch.Tensor:
        extra = self.extra_loss(audio_feats, video_feats, vid_ids)
        intra = self.intra_loss(audio_feats, video_feats, vid_ids)
        return extra + self.lambda_ * intra

    # ======= semantic term  (different videos) ============================
    def extra_loss(
        self,
        audio_feats: torch.Tensor,
        video_feats: torch.Tensor,
        vid_ids: torch.Tensor
    ) -> torch.Tensor:
        # positives are (i,j) where vid_i ≠ vid_j
        pos_mask = vid_ids.unsqueeze(0) != vid_ids.unsqueeze(1)   # (B,B)
        return self.clip_loss(audio_feats, video_feats, pos_mask)

    # ======= temporal term  (same video, different segment) ===============
    def intra_loss(
        self,
        audio_feats: torch.Tensor,
        video_feats: torch.Tensor,
        vid_ids: torch.Tensor
    ) -> torch.Tensor:
        eye = torch.eye(len(vid_ids), dtype=torch.bool, device=vid_ids.device)
        pos_mask = (vid_ids.unsqueeze(0) == vid_ids.unsqueeze(1)) & ~eye
        return self.clip_loss(audio_feats, video_feats, pos_mask)