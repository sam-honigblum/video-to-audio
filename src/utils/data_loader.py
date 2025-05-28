# =============================================================================
# File: src/utils/data_loader.py
# Role: Dataset + DataLoader utilities
# -----------------------------------------------------------------------------

#Write FoleyDataset that returns a dict with:
#{ "mel": FloatTensor(B, 1, F, T),
#  "video": FloatTensor(B, C, Fps, H, W) }

#Add on-the-fly mel conversion (use torchaudio.transforms.MelSpectrogram).

#Verify that a single next(iter(loader)) runs on both CPU and GPU.

# =============================================================================
