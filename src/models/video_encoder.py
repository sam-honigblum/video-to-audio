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

class CAVP(nn.Module):
    def __init__(self):
        super().__init__()

        self.video_encoder = torch.load()

    def forward(self, x):
        pass

class CAVP_Loss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x , lmbda=1.0):
        # return extra_loss + (lambda=1) * intra_loss
        pass

    def extra_loss(): # comparison to other videos
        pass

    def intra_loss(): # comparison to itself in temporal sense
        pass