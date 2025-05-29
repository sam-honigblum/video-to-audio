# =============================================================================
# File: src/infer.py
# Role: Generate Foley audio for a silent video
# -----------------------------------------------------------------------------
# 1. Load checkpoint, frozen encoders, and sampler.
# 2. Extract CAVP tokens from input video.
# 3. Sample noise z_T → denoise with DDIM/DPM-Solver + CFG.
# 4. Decode latent → mel → WAV, then save (and optionally mux to MP4).
# =============================================================================
