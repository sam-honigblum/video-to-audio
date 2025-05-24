# =============================================================================
# File: src/models/sampler.py
# Role: Pluggable samplers (DDIM, DPM-Solver) for inference
# -----------------------------------------------------------------------------
# Classes
#   SamplerBase      : scheduler buffers + utilities
#   DDIMSampler      : deterministic η = 0 update rule
#   DPMSolverSampler : higher-order ODE solver (15-30 steps)
# API
#   sample(model, shape, cond, steps, cfg_scale) → latent z_0
# =============================================================================
