# =============================================================================
# File: src/models/audio_autoencoder.py
# Role: KL VAE that compresses a mel-spectrogram into a latent tensor and decodes it
# -----------------------------------------------------------------------------
# Encoder : Conv ↓, ResBlocks, GroupNorm → (μ, logσ) → reparameterise → z
# Decoder : z → ConvTranspose ↑, ResBlocks → reconstructed mel
# Loss    : L1/L2 recon + β KL
# Public  : encode(mel) | decode(z) | forward(mel) → mel_hat, kl_loss
# =============================================================================
