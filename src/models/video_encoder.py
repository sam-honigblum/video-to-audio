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

from .cavp_modules import Cnn14, ResNet3dSlowOnly

class CAVP(nn.Module):
    def __init__(self, feat_dim=512, temperature=0.07):
        super().__init__()

        self.video_encoder = ResNet3dSlowOnly(depth=50)
        self.video_projection = nn.Linear(2048, feat_dim)
        self.video_temporal_pool = nn.MaxPool1d(kernel_size=16)

        self.audio_encoder = Cnn14(feat_dim=feat_dim)
        self.audio_temporal_pool = nn.MaxPool1d(kernel_size=16)

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

    def forward(self, video, spectrogram):
        """
        video: (B, 3, T, H, W)
        spectrogram: (B, 1, mel_num, T)
        """
        # video encode
        video_feat = self.video_encoder(video)
        b, c, t, h, w = video_feat.shape
        video_feat = video_feat.reshape(b, c, t).permute(0, 2, 1)
        video_feat = self.video_projection(video_feat)
        video_feat = self.video_temporal_pool(video_feat.permute(0, 2, 1)).squeeze(-1)
        video_norm = F.normalize(video_feat, dim=-1)

        # audio encode
        spectrogram = spectrogram.permute(0, 1, 3, 2) # (B, 1, T, mel_num)
        spectrogram_feat = self.audio_encoder(spectrogram) #(B, T, C)
        spectrogram_feat = self.audio_temporal_pool(spectrogram_feat.permute(0, 2, 1))
        spectrogram_norm = F.normalize(spectrogram_feat, dim=-1)

        return video_norm, spectrogram_norm

class _CLIPStyleLoss(nn.Module):
    def __init__(self, shared_logit_scale: nn.Parameter):
        super().__init__()
        self.logit_scale = shared_logit_scale


    def forward(
        self,
        audio: torch.Tensor,             # (B, D)
        video: torch.Tensor,             # (B, D)
        positive_mask: torch.Tensor,      # (B, B)  True ⇒ (i,j) is a positive
    ) -> torch.Tensor:
        B, D = audio.shape
        # 1. l2-normalise
        a = F.normalize(audio, dim=-1)
        v = F.normalize(video, dim=-1)

        # 2. similarity · τ⁻¹
        logits = (self.logit_scale.exp().clamp(max=100) *
                   a @ v.T)                       # (B, B)

        # 3. mask out *unwanted* positives by setting them to −∞
        pos_index_row = positive_mask.float().argmax(dim=1)   # (B,)
        pos_index_col = positive_mask.float().argmax(dim=0)   # (B,)

        loss_a2v = F.cross_entropy(logits,     pos_index_row)
        loss_v2a = F.cross_entropy(logits.T,   pos_index_col)
        return 0.5 * (loss_a2v + loss_v2a)

class CAVP_Loss(nn.Module):
    """
    Implements   L_total = L_extra + λ · L_intra
    where
        L_extra  - semantic contrast  (different videos)
        L_intra  - temporal contrast  (other segments of same video)
    """
    def __init__(self, shared_logit_scale: nn.Parameter, lambda_: float = 1.0):
        super().__init__()
        self.lambda_   = lambda_
        self.clip_loss = _CLIPStyleLoss(shared_logit_scale)

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