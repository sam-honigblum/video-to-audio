# =============================================================================
# File: src/main.py
# Role: Single CLI entry-point for training, inference, and evaluation
# -----------------------------------------------------------------------------
# 1. Merge YAML configs with OmegaConf, then seed RNGs.
# 2. Dispatch to train.py or infer.py based on the first sub-command.
# 3. Support multi-GPU launch (torchrun) and experiment logging.
# =============================================================================
