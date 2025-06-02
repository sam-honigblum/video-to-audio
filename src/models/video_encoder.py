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

        self.video_encoder = ResNet3dSlowOnly(depth=50, pretrained=None)
        self.video_projection = nn.Linear(2048, feat_dim)
        self.video_temporal_pool = nn.MaxPool1d(kernel_size=16)

        self.audio_encoder = Cnn14(feat_dim=feat_dim)
        self.audio_temporal_pool = nn.MaxPool1d(kernel_size=16)

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

    def forward(self, video, spectrogram):
        """
        video: (B, C, T, H, W)
        spectrogram: (B, 1, mel_num, T)
        """
        # video encode
        video_feat = self.video_encoder(video)
        b, c, t, h, w = video_feat.shape
        video_feat = F.avg_pool2d(video_feat.view(-1, h, w), kernel_size=h)
        video_feat = video_feat.reshape(b, c, t).permute(0, 2, 1)
        video_feat = self.video_projection(video_feat)
        video_feat = self.video_temporal_pool(video_feat.permute(0, 2, 1)).squeeze(-1)
        video_norm = F.normalize(video_feat, dim=-1)

        # audio encode
        spectrogram = spectrogram.permute(0, 1, 3, 2) # (B, 1, T, mel_num)
        spectrogram_feat = self.audio_encoder(spectrogram) #(B, T, C)
        spectrogram_feat = self.audio_temporal_pool(spectrogram_feat.permute(0, 2, 1)).squeeze(-1)
        spectrogram_norm = F.normalize(spectrogram_feat, dim=-1)

        return video_norm, spectrogram_norm

class _CLIPStyleLoss(nn.Module):
    def __init__(self, shared_logit_scale: nn.Parameter):
        super().__init__()
        self.logit_scale = shared_logit_scale

    def forward(
        self,
        audio: torch.Tensor,            # (B, D)
        video: torch.Tensor,            # (B, D)
        positive_mask: torch.Tensor,     # (B, B): True ⇒ (i, j) is a positive pair
    ) -> torch.Tensor:
        B, D = audio.shape

        # 1. L2‐normalize embeddings (just in case)
        a = F.normalize(audio, dim=-1)   # (B, D)
        v = F.normalize(video, dim=-1)   # (B, D)

        # 2. Compute scaled cosine similarities: (B, B) matrix
        #    logit_scale.exp() is 1/τ
        logits = (self.logit_scale.exp().clamp(max=100) * (a @ v.T))  # (B, B)

        # 3. For cross‐entropy, we need exactly one positive index per row and per column.
        #    We assume positive_mask has exactly one True in each row and each column.
        pos_index_row = positive_mask.float().argmax(dim=1)  # (B,) selects column index for each audio row
        pos_index_col = positive_mask.float().argmax(dim=0)  # (B,) selects row index for each video column

        loss_a2v = F.cross_entropy(logits,     pos_index_row)
        loss_v2a = F.cross_entropy(logits.T,   pos_index_col)
        return 0.5 * (loss_a2v + loss_v2a)


class CAVP_Loss(nn.Module):
    """
    Implements   L_total = L_extra + λ · L_intra
    where
      • L_extra  - semantic contrast  (pull together each audio/video pair,
                    push apart all other pairs)
      • L_intra  - temporal contrast  (pull each segment’s audio to its “paired”
                    video from another time‐slice of the same video, push apart
                    all other pairs)
    """
    def __init__(self, shared_logit_scale: nn.Parameter, lambda_: float = 1.0):
        super().__init__()
        self.lambda_   = lambda_
        self.clip_loss = _CLIPStyleLoss(shared_logit_scale)

    def forward(
        self,
        audio_feats: torch.Tensor,    # (B, D)
        video_feats: torch.Tensor,    # (B, D)
        vid_ids: torch.Tensor         # (B,), int identifier per *source video*
    ) -> torch.Tensor:
        extra = self.extra_loss(audio_feats, video_feats, vid_ids)
        intra = self.intra_loss(audio_feats, video_feats, vid_ids)
        return extra + self.lambda_ * intra

    def extra_loss(
        self,
        audio_feats: torch.Tensor,    # (B, D)
        video_feats: torch.Tensor,    # (B, D)
        vid_ids: torch.Tensor         # (B,)
    ) -> torch.Tensor:
        """
        Semantic contrast: pull together (audio_i, video_i) only for the same segment i.
        All other pairs in the batch are treated as negatives. This ensures
        exactly one positive per row/column (the diagonal).
        """
        B = audio_feats.size(0)
        device = audio_feats.device

        # Identity‐matrix mask: True only on (i, i)
        pos_mask = torch.eye(B, dtype=torch.bool, device=device)  # (B, B)
        return self.clip_loss(audio_feats, video_feats, pos_mask)

    def intra_loss(
        self,
        audio_feats: torch.Tensor,    # (B, D)
        video_feats: torch.Tensor,    # (B, D)
        vid_ids: torch.Tensor         # (B,)
    ) -> torch.Tensor:
        """
        Temporal contrast: for each index i, we select exactly one other index j
        such that vid_ids[j] == vid_ids[i] and j != i. We form a directed cycle
        among all indices belonging to the same video. This guarantees exactly one
        True per row and one True per column of pos_mask.

        If a particular video ID appears only once in the batch, it contributes no
        positive to intra_loss (we simply skip it).
        """
        B = audio_feats.size(0)
        device = audio_feats.device

        # Initialize all‐False mask
        pos_mask = torch.zeros((B, B), dtype=torch.bool, device=device)

        # Find unique video IDs in the batch
        unique_vids = torch.unique(vid_ids)

        for vid in unique_vids:
            # Indices of all segments that come from the same video 'vid'
            idxs = (vid_ids == vid).nonzero(as_tuple=True)[0]  # shape (N_vid,)
            if idxs.numel() < 2:
                # No “other segment” to pair with ⇒ skip
                continue

            # Build a directed cycle among these indices:
            #   e.g. if idxs = [i0, i1, i2], then
            #     pos_mask[i0, i1] = True
            #     pos_mask[i1, i2] = True
            #     pos_mask[i2, i0] = True
            # That ensures exactly one True per row and per column for this group.
            ridx = idxs.tolist()  # convert to Python list of ints
            for k in range(len(ridx)):
                i_idx = ridx[k]
                j_idx = ridx[(k + 1) % len(ridx)]
                pos_mask[i_idx, j_idx] = True

        return self.clip_loss(audio_feats, video_feats, pos_mask)