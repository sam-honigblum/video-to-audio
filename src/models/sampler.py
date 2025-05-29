# =============================================================================
# File: src/models/sampler.py
# Role: DPM‑Solver sampler wrapper for latent‑diffusion inference
# -----------------------------------------------------------------------------
# • Implements Lu et al.'s multistep **DPM‑Solver++** only (no DDIM fallback).
# • Keeps an alias `ddim_sample()` so legacy calls stay valid.
# • Relies on 🤗 diffusers ≥ 0.25 which already ships the scheduler.
# =============================================================================

from __future__ import annotations

from typing import Optional

import torch
from diffusers import DPMSolverMultistepScheduler


class DPMSolverSampler:
    """Thin wrapper around *diffusers* multistep DPM‑Solver scheduler.

    Parameters
    ----------
    diffusion : LatentDiffusion
        The trained LatentDiffusion instance (provides `unet`).
    scheduler : Optional[DPMSolverMultistepScheduler]
        Pre‑constructed scheduler.  If *None*, one is created with the same
        number of training timesteps as `diffusion` and a squared‑cosine β‑schedule.
    """

    def __init__(
        self,
        diffusion,
        scheduler: Optional[DPMSolverMultistepScheduler] = None,
    ) -> None:
        self.diffusion = diffusion

        if scheduler is None:
            # A generic squared‑cosine schedule works for most pretrained LDMs.
            scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=diffusion.alphas_cumprod.shape[0],
                beta_schedule="squaredcos_cap_v2",
                prediction_type="epsilon",
            )
        self.scheduler = scheduler

    # ------------------------------------------------------------------ public

    @torch.no_grad()
    def dpm_sample(
        self,
        x_T: torch.Tensor,
        cond: torch.Tensor,
        steps: int,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Run multistep DPM‑Solver sampling.

        Parameters
        ----------
        x_T : tensor  (B, C, 1, W)
            Pure Gaussian noise latents.
        cond : tensor  (B, 40, 512)
            Video tokens for cross‑attention.
        steps : int
            Number of solver steps (e.g. 20 – 30).
        cfg_scale : float
            Classifier‑free guidance scale (> 1 boosts conditioning strength).
        """
        device = x_T.device
        unet = self.diffusion.unet

        self.scheduler.set_timesteps(steps, device=device)
        x = x_T

        for i, t in enumerate(self.scheduler.timesteps):
            # -------------------------- UNet forward (with optional CFG)
            if cfg_scale == 1.0:
                eps = unet(x, t, cond)
            else:
                eps_cond = unet(x, t, cond)
                eps_uncond = unet(x, t, torch.zeros_like(cond))
                eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

            # -------------------------- scheduler update (prev_sample)
            x = self.scheduler.step(eps, t, x, return_dict=False)

        return x

    # Alias so existing code using `.ddim_sample()` keeps working
    ddim_sample = dpm_sample


# -----------------------------------------------------------------------------
# Convenience factory (optional import style)
# -----------------------------------------------------------------------------

def build_sampler(diffusion) -> DPMSolverSampler:  # noqa: D401
    """Factory that returns a ready‑made DPM‑Solver sampler."""
    return DPMSolverSampler(diffusion)
