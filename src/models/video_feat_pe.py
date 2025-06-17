import torch
import torch.nn as nn

class CAVPVideoPE(nn.Module):

    def __init__(self, original_dim=512, transformed_dim=512, seq_len=40, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project = nn.Sequential(
            nn.Linear(original_dim, transformed_dim),
            nn.ReLU(),
            nn.Linear(transformed_dim, transformed_dim)
        )
        self.pe = nn.Embedding(seq_len, transformed_dim)

    def forward(self, x):
        batch_size, seq_len, latent_dim = x.shape
        x = self.project(x)
        pe = self.pe(torch.arange(seq_len, device=x.device).reshape(1, -1))
        pe = pe.repeat(batch_size, 1, 1)

        return x + pe