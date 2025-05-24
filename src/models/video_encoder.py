# =============================================================================
# File: src/models/video_encoder.py
# Role: Frozen CAVP backbone that maps RGB video to 40×512 tokens
# -----------------------------------------------------------------------------
# • Wraps a pretrained 3D-ViT (TimeSformer) fine-tuned with CAVP loss.
# • Adds positional encodings so cross-attention lines up temporally.
# • forward(video) → (B, 40, 512) for UNet cross-attention.
# =============================================================================
