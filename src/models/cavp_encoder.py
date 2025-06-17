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
    def __init__(self, feat_dim=512, temperature=0.07, pretrain=False):
        super().__init__()

        vid_pretrained = None
        aud_pretrained = None
        if pretrain:
            vid_pretrained = "video-to-audio/pretrained-models/resnet50-slowfast-encoder.pth"
            aud_pretrained = torch.load("video-to-audio/pretrained-models/cnn14-encoder.pth", map_location="cpu")["model"]

        self.video_encoder = ResNet3dSlowOnly(depth=50, pretrained=vid_pretrained)
        self.video_projection = nn.Linear(2048, feat_dim)
        self.video_max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.video_mean_pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.audio_encoder = Cnn14(feat_dim=feat_dim)

        if aud_pretrained:
            self.audio_encoder.load_state_dict(aud_pretrained, strict=False)

        self.audio_max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.audio_mean_pool = nn.AdaptiveAvgPool1d(output_size=1)

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

class CAVP_Loss(nn.Module):
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

        logits_per_vid[~torch.isfinite(logits_per_vid)] = 0.0
        logits_per_aud[~torch.isfinite(logits_per_aud)] = 0.0

        num_logits = logits_per_vid.shape[0]
        labels = torch.arange(num_logits, device=video_feats.device, dtype=torch.long)
        extra_loss = (F.cross_entropy(logits_per_vid, labels) + F.cross_entropy(logits_per_aud, labels)) / 2.0

        batches, dim = video_mean_feats.shape
        vid_intra_feats = video_mean_feats.reshape(-1, self.clip_num, dim)
        aud_intra_feats = audio_mean_feats.reshape(-1, self.clip_num, dim)

        intra_logits_per_vid = logit_scale * torch.matmul(vid_intra_feats, aud_intra_feats.permute(0, 2, 1))
        intra_logits_per_aud = logit_scale * torch.matmul(aud_intra_feats, vid_intra_feats.permute(0, 2, 1))

        intra_logits_per_vid[~torch.isfinite(intra_logits_per_vid)] = 0.0
        intra_logits_per_aud[~torch.isfinite(intra_logits_per_aud)] = 0.0

        intra_batches, intra_num_logits, _ = intra_logits_per_vid.shape
        intra_logits_per_vid = intra_logits_per_vid.reshape(intra_batches * intra_num_logits, intra_num_logits)
        intra_logits_per_aud = intra_logits_per_aud.reshape(intra_batches * intra_num_logits, intra_num_logits)

        intra_labels = torch.arange(intra_num_logits, device=video_mean_feats.device, dtype=torch.long).unsqueeze(0).repeat(intra_batches, 1).flatten() # create labels for everything all batches
        intra_loss = (F.cross_entropy(intra_logits_per_vid, intra_labels) + F.cross_entropy(intra_logits_per_aud, intra_labels)) / 2.0

        return extra_loss + self.lambda_ * intra_loss


class CAVP_VideoOnly(nn.Module):
    def __init__(self, ckpt:str, feat_dim:int=512):
        super().__init__()
        self.backbone = CAVP(feat_dim=feat_dim)
        
        # Load checkpoint
        checkpoint = torch.load(ckpt, map_location="cpu")
        state_dict = checkpoint["model"]
        
        # Create key mapping for audio encoder
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("audio_encoder."):
                # Map old keys to new keys
                if "bn0" in k:
                    new_key = k.replace("bn0", "bn")
                elif "fc_audioset" in k:
                    new_key = k.replace("fc_audioset", "project_head")
                else:
                    new_key = k
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        
        # Load state dict with strict=False to handle any remaining mismatches
        self.backbone.load_state_dict(new_state_dict, strict=False)
        self.backbone.eval().requires_grad_(False)
    
    @torch.no_grad()
    def forward(self, video):
        v_max, v_mean, *_ = self.backbone(video, torch.empty(0))  # spectro factice
        return v_mean           #  vector 512-D

