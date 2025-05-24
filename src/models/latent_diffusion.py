# =============================================================================
# File: src/models/latent_diffusion.py
# Role: UNet-based latent diffusion core with video cross-attention
# -----------------------------------------------------------------------------
# • UNet backbone: attention at res 4, 2, 1.
# • Cross-Attention injects video tokens after every block.
# • register_schedule() pre-computes β_t, ᾱ_t tables.
# • p_losses(), p_sample_loop() implement DDPM training + double guidance hook.
# =============================================================================
