# video-to-audio


## What lives in `src/models/` and why

| File | Role | Key I/O |
|------|------|---------|
| **audio_autoencoder.py** | Compresses each log-mel spectrogram into a small latent tensor **z** and decodes it back. The encoder trains once; the decoder is frozen during diffusion. | `encode(x) → z`<br>`decode(z) → x̂` |
| **video_encoder.py** | Turns a stack of video frames into 40×512 CAVP features that carry “what/when” cues for sound. | `encode(frames) → feats` |
| **latent_diffusion.py** | Core UNet that predicts the noise to remove at every timestep. Internally hosts the β-scheduler and cross-attention layers that inject the video features. | `forward(z_t, t, feats) → ε̂_t` |
| **sampler.py** | Implements DDIM / DPM-Solver loops. Starts from pure Gaussian noise **z_T** and repeatedly calls `latent_diffusion.forward` until **z₀** is recovered. | `sample(model, feats, steps) → z₀` |

### How the whole pipeline runs

1. **Encode** `z₀ = AudioAutoencoder.encode(spectrogram)` → **16× smaller** latent space for efficiency.
2. **Visual cues** `v = VideoEncoder.encode(frames)` → time-aligned CAVP tokens.
3. **Diffusion** Add noise → `z_t`. The UNet denoises step-by-step, guided by cross-attention on **v**.
4. **Sample** `ẑ₀ = Sampler.sample(LatentDiffusion, v, N_steps)`
5. **Decode** `spectrogram̂ = AudioAutoencoder.decode(ẑ₀)` → inverse STFT → final WAV, now in sync with the video.

That’s the entire Diff-Foley loop in fewer than 50 lines of glue code.

