python -m src.infer \
       --video samples/silent.mp4 \
       --config configs/infer.yaml \
       --checkpoint checkpoints/best.pt \
       --out_wav output.wav \
       --mux         # optional
