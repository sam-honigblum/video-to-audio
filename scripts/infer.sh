#!/usr/bin/env bash
# ---------------------------------------------------------
# inference helper for video-to-audio repo
# ---------------------------------------------------------
# USAGE
#   ./infer.sh <video.mp4> [output-stem] [extra flags passed to infer.py]
#
# EXAMPLES
#   ./infer.sh data/cat.mp4                 # → cat_foley.wav (+mp4 if --mux)
#   ./infer.sh data/cat.mp4 mytrack -s 80   # → mytrack.wav, 80 sampling steps
#
# The script wires default checkpoint & config paths so you can simply point
# it at a video. Override them via environment variables or CLI flags.
#   LDM_CKPT   (default: checkpoints/ldm/ldm-step4500.pt)
#   CAVP_CKPT  (default: checkpoints/cavp/cavp-7500steps.pt)
#   CONFIG     (default: configs/infer.yaml)
#
# Any additional flags after the first two positional args are forwarded
# unchanged to infer.py (e.g. --steps, --guidance, --mux).

set -euo pipefail

VIDEO=${1:?"First argument must be a video file"}
shift || true

STEM="${1:-${VIDEO##*/}}"   # second arg optional output‑stem
STEM="${STEM%.*}"
[[ $# -gt 0 ]] && shift || true  # drop stem param if present

# Default paths — override via env or CLI
LDM_CKPT="${LDM_CKPT:-checkpoints/ldm/ldm-step4500.pt}"
CAVP_CKPT="${CAVP_CKPT:-checkpoints/cavp/cavp-7500steps.pt}"
CONFIG="${CONFIG:-configs/infer.yaml}"

python infer.py \
  --video       "${VIDEO}" \
  --out_stem    "${STEM}" \
  --ldm_ckpt    "${LDM_CKPT}" \
  --cavp_ckpt   "${CAVP_CKPT}" \
  --config      "${CONFIG}" \
  "$@"
