from diffusers import UNet2DConditionModel


def build_unet(in_channels: int, model_channels: int = 320) -> nn.Module:
    """Light wrapper that mirrors Diff‑Foley’s UNet geometry."""
    return UNet2DConditionModel(
        sample_size=(1, 32),                 # (height, width) – height is dummy (1)
        in_channels=in_channels,             # 8 for EnCodec‑24kHz
        out_channels=in_channels,            # predict ε with same channels
        layers_per_block=2,
        block_out_channels=(model_channels, model_channels * 2, model_channels * 4),
        down_block_types=(
            "DownBlock2D",      # 1/2 W
            "AttnDownBlock2D",  # 1/4 W + self‑attention
            "AttnDownBlock2D",  # 1/8 W
        ),
        up_block_types=(
            "AttnUpBlock2D",    # 1/4 W
            "AttnUpBlock2D",    # 1/2 W
            "UpBlock2D",        # full W
        ),
        cross_attention_dim=512,            # matches CAVP token size
    )
