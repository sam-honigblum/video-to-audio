# =============================================================================
# File: src/train.py
# Role: End-to-end training loop for the Diff-Foley Latent Diffusion Model
# -----------------------------------------------------------------------------
#  ▸ Builds: audio auto-encoder (E/D), frozen CAVP video encoder, UNet, sampler.
#  ▸ Loads paired video–audio data via utils/data_loader.FoleyDataset.
#  ▸ Computes diffusion loss + optional classifier guidance loss.
#  ▸ Maintains an EMA of UNet weights, saves checkpoints, logs metrics.
# =============================================================================
