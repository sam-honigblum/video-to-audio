import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.modules.utils import _ntuple, _triple

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from mmcv.cnn import ConvModule, NonLocal3d, build_activation_layer, kaiming_init
from mmcv.utils import print_log
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, Sequential
from mmengine.model.weight_init import constant_init, kaiming_init
from mmengine.runner import _load_checkpoint, load_checkpoint
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

"""
PANN CNN14 Audio Encoder: https://github.com/qiuqiangkong/audioset_tagging_cnn
"""


def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes
    (1, 3, 5, ...).

    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)

    Returns:
      out: (batch_size, ...)
    """
    out = (
        x[0::2].transpose(0, -1) * mixup_lambda[0::2]
        + x[1::2].transpose(0, -1) * mixup_lambda[1::2]
    ).transpose(0, -1)
    return out


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class Cnn14(nn.Module):
    def __init__(
        self, feat_dim=512
    ):

        super(Cnn14, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, feat_dim, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        lat_x = x1 + x2
        lat_x = lat_x.transpose(1, 2)
        lat_x = F.relu_(self.fc1(lat_x))
        x = F.relu_(self.fc1(lat_x))
        output = self.final_project(x)
        return output


"""
SlowFast Video Encoder: https://github.com/open-mmlab/mmaction2
"""


class BasicBlock3d(BaseModule):
    """BasicBlock 3d block for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        spatial_stride (int): Spatial stride in the conv3d layer.
            Defaults to 1.
        temporal_stride (int): Temporal stride in the conv3d layer.
            Defaults to 1.
        dilation (int): Spacing between kernel elements. Defaults to 1.
        downsample (nn.Module or None): Downsample layer. Defaults to None.
        style (str): 'pytorch' or 'caffe'. If set to 'pytorch', the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Defaults to ``'pytorch'``.
        inflate (bool): Whether to inflate kernel. Defaults to True.
        non_local (bool): Determine whether to apply non-local module in this
            block. Defaults to False.
        non_local_cfg (dict): Config for non-local module.
            Defaults to ``dict()``.
        conv_cfg (dict): Config dict for convolution layer.
            Defaults to ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers.
            Required keys are ``type``. Defaults to ``dict(type='BN3d')``.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='ReLU')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        spatial_stride: int = 1,
        temporal_stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        style: str = "pytorch",
        inflate: bool = True,
        non_local: bool = False,
        non_local_cfg: Dict = dict(),
        conv_cfg: Dict = dict(type="Conv3d"),
        norm_cfg: Dict = dict(type="BN3d"),
        act_cfg: Dict = dict(type="ReLU"),
        with_cp: bool = False,
        init_cfg: Optional[Union[Dict, List[Dict]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert style in ["pytorch", "caffe"]
        # make sure that only ``inflate_style`` is passed into kwargs
        assert set(kwargs).issubset(["inflate_style"])

        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.style = style
        self.inflate = inflate
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.non_local = non_local
        self.non_local_cfg = non_local_cfg

        self.conv1_stride_s = spatial_stride
        self.conv2_stride_s = 1
        self.conv1_stride_t = temporal_stride
        self.conv2_stride_t = 1

        if self.inflate:
            conv1_kernel_size = (3, 3, 3)
            conv1_padding = (1, dilation, dilation)
            conv2_kernel_size = (3, 3, 3)
            conv2_padding = (1, 1, 1)
        else:
            conv1_kernel_size = (1, 3, 3)
            conv1_padding = (0, dilation, dilation)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, 1, 1)

        self.conv1 = ConvModule(
            inplanes,
            planes,
            conv1_kernel_size,
            stride=(self.conv1_stride_t, self.conv1_stride_s, self.conv1_stride_s),
            padding=conv1_padding,
            dilation=(1, dilation, dilation),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        self.conv2 = ConvModule(
            planes,
            planes * self.expansion,
            conv2_kernel_size,
            stride=(self.conv2_stride_t, self.conv2_stride_s, self.conv2_stride_s),
            padding=conv2_padding,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None,
        )

        self.downsample = downsample
        self.relu = build_activation_layer(self.act_cfg)

        if self.non_local:
            self.non_local_block = NonLocal3d(
                self.conv2.norm.num_features, **self.non_local_cfg
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x

            out = self.conv1(x)
            out = self.conv2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)

        if self.non_local:
            out = self.non_local_block(out)

        return out


class Bottleneck3d(BaseModule):
    """Bottleneck 3d block for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        spatial_stride (int): Spatial stride in the conv3d layer.
            Defaults to 1.
        temporal_stride (int): Temporal stride in the conv3d layer.
            Defaults to 1.
        dilation (int): Spacing between kernel elements. Defaults to 1.
        downsample (nn.Module, optional): Downsample layer. Defaults to None.
        style (str): 'pytorch' or 'caffe'. If set to 'pytorch', the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Defaults to ``'pytorch'``.
        inflate (bool): Whether to inflate kernel. Defaults to True.
        inflate_style (str): '3x1x1' or '3x3x3'. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Defaults to ``'3x1x1'``.
        non_local (bool): Determine whether to apply non-local module in this
            block. Defaults to False.
        non_local_cfg (dict): Config for non-local module.
            Defaults to ``dict()``.
        conv_cfg (dict): Config dict for convolution layer.
            Defaults to ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers. required
            keys are ``type``. Defaults to ``dict(type='BN3d')``.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='ReLU')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        spatial_stride: int = 1,
        temporal_stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        style: str = "pytorch",
        inflate: bool = True,
        inflate_style: str = "3x1x1",
        non_local: bool = False,
        non_local_cfg: Dict = dict(),
        conv_cfg: Dict = dict(type="Conv3d"),
        norm_cfg: Dict = dict(type="BN3d"),
        act_cfg: Dict = dict(type="ReLU"),
        with_cp: bool = False,
        init_cfg: Optional[Union[Dict, List[Dict]]] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert style in ["pytorch", "caffe"]
        assert inflate_style in ["3x1x1", "3x3x3"]

        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.style = style
        self.inflate = inflate
        self.inflate_style = inflate_style
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.non_local = non_local
        self.non_local_cfg = non_local_cfg

        if self.style == "pytorch":
            self.conv1_stride_s = 1
            self.conv2_stride_s = spatial_stride
            self.conv1_stride_t = 1
            self.conv2_stride_t = temporal_stride
        else:
            self.conv1_stride_s = spatial_stride
            self.conv2_stride_s = 1
            self.conv1_stride_t = temporal_stride
            self.conv2_stride_t = 1

        if self.inflate:
            if inflate_style == "3x1x1":
                conv1_kernel_size = (3, 1, 1)
                conv1_padding = (1, 0, 0)
                conv2_kernel_size = (1, 3, 3)
                conv2_padding = (0, dilation, dilation)
            else:
                conv1_kernel_size = (1, 1, 1)
                conv1_padding = (0, 0, 0)
                conv2_kernel_size = (3, 3, 3)
                conv2_padding = (1, dilation, dilation)
        else:
            conv1_kernel_size = (1, 1, 1)
            conv1_padding = (0, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, dilation, dilation)

        self.conv1 = ConvModule(
            inplanes,
            planes,
            conv1_kernel_size,
            stride=(self.conv1_stride_t, self.conv1_stride_s, self.conv1_stride_s),
            padding=conv1_padding,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        self.conv2 = ConvModule(
            planes,
            planes,
            conv2_kernel_size,
            stride=(self.conv2_stride_t, self.conv2_stride_s, self.conv2_stride_s),
            padding=conv2_padding,
            dilation=(1, dilation, dilation),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        self.conv3 = ConvModule(
            planes,
            planes * self.expansion,
            1,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            # No activation in the third ConvModule for bottleneck
            act_cfg=None,
        )

        self.downsample = downsample
        self.relu = build_activation_layer(self.act_cfg)

        if self.non_local:
            self.non_local_block = NonLocal3d(
                self.conv3.norm.num_features, **self.non_local_cfg
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x

            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)

        if self.non_local:
            out = self.non_local_block(out)

        return out


class DeConvModule(BaseModule):
    """A deconv module that bundles deconv/norm/activation layers.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
        stride (int | tuple[int]): Stride of the convolution.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input.
        bias (bool): Whether to add a learnable bias to the output.
            Defaults to False.
        with_bn (bool): Whether to add a BN layer. Defaults to True.
        with_relu (bool): Whether to add a ReLU layer. Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Union[int, Tuple[int]] = (1, 1, 1),
        padding: Union[int, Tuple[int]] = 0,
        bias: bool = False,
        with_bn: bool = True,
        with_relu: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.with_bn = with_bn
        self.with_relu = with_relu

        self.conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        # x should be a 5-d tensor
        assert len(x.shape) == 5
        N, C, T, H, W = x.shape
        out_shape = (
            N,
            self.out_channels,
            self.stride[0] * T,
            self.stride[1] * H,
            self.stride[2] * W,
        )
        x = self.conv(x, output_size=out_shape)
        if self.with_bn:
            x = self.bn(x)
        if self.with_relu:
            x = self.relu(x)
        return x


class ResNet3d(BaseModule):
    """ResNet 3d backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
            Defaults to 50.
        pretrained (str, optional): Name of pretrained model. Defaults to None.
        stage_blocks (tuple, optional): Set number of stages for each res
            layer. Defaults to None.
        pretrained2d (bool): Whether to load pretrained 2D model.
            Defaults to True.
        in_channels (int): Channel num of input features. Defaults to 3.
        num_stages (int): Resnet stages. Defaults to 4.
        base_channels (int): Channel num of stem output features.
            Defaults to 64.
        out_indices (Sequence[int]): Indices of output feature.
            Defaults to ``(3, )``.
        spatial_strides (Sequence[int]):
            Spatial strides of residual blocks of each stage.
            Defaults to ``(1, 2, 2, 2)``.
        temporal_strides (Sequence[int]):
            Temporal strides of residual blocks of each stage.
            Defaults to ``(1, 1, 1, 1)``.
        dilations (Sequence[int]): Dilation of each stage.
            Defaults to ``(1, 1, 1, 1)``.
        conv1_kernel (Sequence[int]): Kernel size of the first conv layer.
            Defaults to ``(3, 7, 7)``.
        conv1_stride_s (int): Spatial stride of the first conv layer.
            Defaults to 2.
        conv1_stride_t (int): Temporal stride of the first conv layer.
            Defaults to 1.
        pool1_stride_s (int): Spatial stride of the first pooling layer.
            Defaults to 2.
        pool1_stride_t (int): Temporal stride of the first pooling layer.
            Defaults to 1.
        with_pool2 (bool): Whether to use pool2. Defaults to True.
        style (str): 'pytorch' or 'caffe'. If set to 'pytorch', the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Defaults to ``'pytorch'``.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Defaults to -1.
        inflate (Sequence[int]): Inflate Dims of each block.
            Defaults to ``(1, 1, 1, 1)``.
        inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Defaults to ``3x1x1``.
        conv_cfg (dict): Config for conv layers.
            Required keys are ``type``. Defaults to ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers.
            Required keys are ``type`` and ``requires_grad``.
            Defaults to ``dict(type='BN3d', requires_grad=True)``.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='ReLU', inplace=True)``.
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (``mean`` and ``var``). Defaults to False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        non_local (Sequence[int]): Determine whether to apply non-local module
            in the corresponding block of each stages.
            Defaults to ``(0, 0, 0, 0)``.
        non_local_cfg (dict): Config for non-local module.
            Defaults to ``dict()``.
        zero_init_residual (bool):
            Whether to use zero initialization for residual block,
            Defaults to True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    arch_settings = {
        18: (BasicBlock3d, (2, 2, 2, 2)),
        34: (BasicBlock3d, (3, 4, 6, 3)),
        50: (Bottleneck3d, (3, 4, 6, 3)),
        101: (Bottleneck3d, (3, 4, 23, 3)),
        152: (Bottleneck3d, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        depth: int = 50,
        pretrained: Optional[str] = None,
        stage_blocks: Optional[Tuple] = None,
        pretrained2d: bool = True,
        in_channels: int = 3,
        num_stages: int = 4,
        base_channels: int = 64,
        out_indices: Sequence[int] = (3,),
        spatial_strides: Sequence[int] = (1, 2, 2, 2),
        temporal_strides: Sequence[int] = (1, 1, 1, 1),
        dilations: Sequence[int] = (1, 1, 1, 1),
        conv1_kernel: Sequence[int] = (3, 7, 7),
        conv1_stride_s: int = 2,
        conv1_stride_t: int = 1,
        pool1_stride_s: int = 2,
        pool1_stride_t: int = 1,
        with_pool1: bool = True,
        with_pool2: bool = True,
        style: str = "pytorch",
        frozen_stages: int = -1,
        inflate: Sequence[int] = (1, 1, 1, 1),
        inflate_style: str = "3x1x1",
        conv_cfg: Dict = dict(type="Conv3d"),
        norm_cfg: Dict = dict(type="BN3d", requires_grad=True),
        act_cfg: Dict = dict(type="ReLU", inplace=True),
        norm_eval: bool = False,
        with_cp: bool = False,
        non_local: Sequence[int] = (0, 0, 0, 0),
        non_local_cfg: Dict = dict(),
        zero_init_residual: bool = True,
        init_cfg: Optional[Union[Dict, List[Dict]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if depth not in self.arch_settings:
            raise KeyError(f"invalid depth {depth} for resnet")
        self.depth = depth
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.stage_blocks = stage_blocks
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert (
            len(spatial_strides)
            == len(temporal_strides)
            == len(dilations)
            == num_stages
        )
        if self.stage_blocks is not None:
            assert len(self.stage_blocks) == num_stages

        self.conv1_kernel = conv1_kernel
        self.conv1_stride_s = conv1_stride_s
        self.conv1_stride_t = conv1_stride_t
        self.pool1_stride_s = pool1_stride_s
        self.pool1_stride_t = pool1_stride_t
        self.with_pool1 = with_pool1
        self.with_pool2 = with_pool2
        self.style = style
        self.frozen_stages = frozen_stages
        self.stage_inflations = _ntuple(num_stages)(inflate)
        self.non_local_stages = _ntuple(num_stages)(non_local)
        self.inflate_style = inflate_style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        self.block, stage_blocks = self.arch_settings[depth]

        if self.stage_blocks is None:
            self.stage_blocks = stage_blocks[:num_stages]

        self.inplanes = self.base_channels

        self.non_local_cfg = non_local_cfg

        self._make_stem_layer()

        self.res_layers = []
        lateral_inplanes = getattr(self, "lateral_inplanes", [0, 0, 0, 0])

        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = self.base_channels * 2**i
            res_layer = self.make_res_layer(
                self.block,
                self.inplanes + lateral_inplanes[i],
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                style=self.style,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
                act_cfg=self.act_cfg,
                non_local=self.non_local_stages[i],
                non_local_cfg=self.non_local_cfg,
                inflate=self.stage_inflations[i],
                inflate_style=self.inflate_style,
                with_cp=with_cp,
                **kwargs,
            )
            self.inplanes = planes * self.block.expansion
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = (
            self.block.expansion
            * self.base_channels
            * 2 ** (len(self.stage_blocks) - 1)
        )

    @staticmethod
    def make_res_layer(
        block: nn.Module,
        inplanes: int,
        planes: int,
        blocks: int,
        spatial_stride: Union[int, Sequence[int]] = 1,
        temporal_stride: Union[int, Sequence[int]] = 1,
        dilation: int = 1,
        style: str = "pytorch",
        inflate: Union[int, Sequence[int]] = 1,
        inflate_style: str = "3x1x1",
        non_local: Union[int, Sequence[int]] = 0,
        non_local_cfg: Dict = dict(),
        norm_cfg: Optional[Dict] = None,
        act_cfg: Optional[Dict] = None,
        conv_cfg: Optional[Dict] = None,
        with_cp: bool = False,
        **kwargs,
    ) -> nn.Module:
        """Build residual layer for ResNet3D.

        Args:
            block (nn.Module): Residual module to be built.
            inplanes (int): Number of channels for the input feature
                in each block.
            planes (int): Number of channels for the output feature
                in each block.
            blocks (int): Number of residual blocks.
            spatial_stride (int | Sequence[int]): Spatial strides in
                residual and conv layers. Defaults to 1.
            temporal_stride (int | Sequence[int]): Temporal strides in
                residual and conv layers. Defaults to 1.
            dilation (int): Spacing between kernel elements. Defaults to 1.
            style (str): 'pytorch' or 'caffe'. If set to 'pytorch', the
                stride-two layer is the 3x3 conv layer,otherwise the
                stride-two layer is the first 1x1 conv layer.
                Defaults to ``'pytorch'``.
            inflate (int | Sequence[int]): Determine whether to inflate
                for each block. Defaults to 1.
            inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines
                the kernel sizes and padding strides for conv1 and conv2
                in each block. Default: ``'3x1x1'``.
            non_local (int | Sequence[int]): Determine whether to apply
                non-local module in the corresponding block of each stages.
                Defaults to 0.
            non_local_cfg (dict): Config for non-local module.
                Defaults to ``dict()``.
            conv_cfg (dict, optional): Config for conv layers.
                Defaults to None.
            norm_cfg (dict, optional): Config for norm layers.
                Defaults to None.
            act_cfg (dict, optional): Config for activate layers.
                Defaults to None.
            with_cp (bool, optional): Use checkpoint or not. Using checkpoint
                will save some memory while slowing down the training speed.
                Defaults to False.

        Returns:
            nn.Module: A residual layer for the given config.
        """
        inflate = inflate if not isinstance(inflate, int) else (inflate,) * blocks
        non_local = (
            non_local if not isinstance(non_local, int) else (non_local,) * blocks
        )
        assert len(inflate) == blocks and len(non_local) == blocks
        downsample = None
        if spatial_stride != 1 or inplanes != planes * block.expansion:
            downsample = ConvModule(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
            )

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                downsample=downsample,
                style=style,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                non_local=(non_local[0] == 1),
                non_local_cfg=non_local_cfg,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp,
                **kwargs,
            )
        )
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    spatial_stride=1,
                    temporal_stride=1,
                    dilation=dilation,
                    style=style,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    non_local=(non_local[i] == 1),
                    non_local_cfg=non_local_cfg,
                    norm_cfg=norm_cfg,
                    conv_cfg=conv_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp,
                    **kwargs,
                )
            )

        return Sequential(*layers)

    @staticmethod
    def _inflate_conv_params(
        conv3d: nn.Module,
        state_dict_2d: OrderedDict,
        module_name_2d: str,
        inflated_param_names: List[str],
    ) -> None:
        """Inflate a conv module from 2d to 3d.

        Args:
            conv3d (nn.Module): The destination conv3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding conv module in the
                2d model.
            inflated_param_names (list[str]): List of parameters that have been
                inflated.
        """
        weight_2d_name = module_name_2d + ".weight"

        conv2d_weight = state_dict_2d[weight_2d_name]
        kernel_t = conv3d.weight.data.shape[2]

        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)

        if getattr(conv3d, "bias") is not None:
            bias_2d_name = module_name_2d + ".bias"
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)

    @staticmethod
    def _inflate_bn_params(
        bn3d: nn.Module,
        state_dict_2d: OrderedDict,
        module_name_2d: str,
        inflated_param_names: List[str],
    ) -> None:
        """Inflate a norm module from 2d to 3d.

        Args:
            bn3d (nn.Module): The destination bn3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding bn module in the
                2d model.
            inflated_param_names (list[str]): List of parameters that have been
                inflated.
        """
        for param_name, param in bn3d.named_parameters():
            param_2d_name = f"{module_name_2d}.{param_name}"
            param_2d = state_dict_2d[param_2d_name]
            if param.data.shape != param_2d.shape:
                warnings.warn(
                    f"The parameter of {module_name_2d} is not"
                    "loaded due to incompatible shapes. "
                )
                return

            param.data.copy_(param_2d)
            inflated_param_names.append(param_2d_name)

        for param_name, param in bn3d.named_buffers():
            param_2d_name = f"{module_name_2d}.{param_name}"
            # some buffers like num_batches_tracked may not exist in old
            # checkpoints
            if param_2d_name in state_dict_2d:
                param_2d = state_dict_2d[param_2d_name]
                param.data.copy_(param_2d)
                inflated_param_names.append(param_2d_name)

    @staticmethod
    def _inflate_weights(self, logger: MMLogger) -> None:
        """Inflate the resnet2d parameters to resnet3d.

        The differences between resnet3d and resnet2d mainly lie in an extra
        axis of conv kernel. To utilize the pretrained parameters in 2d model,
        the weight of conv2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (MMLogger): The logger used to print
                debugging information.
        """

        state_dict_r2d = _load_checkpoint(self.pretrained, map_location="cpu")
        if "state_dict" in state_dict_r2d:
            state_dict_r2d = state_dict_r2d["state_dict"]

        inflated_param_names = []
        for name, module in self.named_modules():
            if isinstance(module, ConvModule):
                # we use a ConvModule to wrap conv+bn+relu layers, thus the
                # name mapping is needed
                if "downsample" in name:
                    # layer{X}.{Y}.downsample.conv->layer{X}.{Y}.downsample.0
                    original_conv_name = name + ".0"
                    # layer{X}.{Y}.downsample.bn->layer{X}.{Y}.downsample.1
                    original_bn_name = name + ".1"
                else:
                    # layer{X}.{Y}.conv{n}.conv->layer{X}.{Y}.conv{n}
                    original_conv_name = name
                    # layer{X}.{Y}.conv{n}.bn->layer{X}.{Y}.bn{n}
                    original_bn_name = name.replace("conv", "bn")
                if original_conv_name + ".weight" not in state_dict_r2d:
                    logger.warning(
                        f"Module not exist in the state_dict_r2d"
                        f": {original_conv_name}"
                    )
                else:
                    shape_2d = state_dict_r2d[original_conv_name + ".weight"].shape
                    shape_3d = module.conv.weight.data.shape
                    if shape_2d != shape_3d[:2] + shape_3d[3:]:
                        logger.warning(
                            f"Weight shape mismatch for "
                            f": {original_conv_name} : "
                            f"3d weight shape: {shape_3d}; "
                            f"2d weight shape: {shape_2d}. "
                        )
                    else:
                        self._inflate_conv_params(
                            module.conv,
                            state_dict_r2d,
                            original_conv_name,
                            inflated_param_names,
                        )

                if original_bn_name + ".weight" not in state_dict_r2d:
                    logger.warning(
                        f"Module not exist in the state_dict_r2d"
                        f": {original_bn_name}"
                    )
                else:
                    self._inflate_bn_params(
                        module.bn,
                        state_dict_r2d,
                        original_bn_name,
                        inflated_param_names,
                    )

        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(state_dict_r2d.keys()) - set(inflated_param_names)
        if remaining_names:
            logger.info(
                f"These parameters in the 2d checkpoint are not loaded"
                f": {remaining_names}"
            )

    def inflate_weights(self, logger: MMLogger) -> None:
        """Inflate weights."""
        self._inflate_weights(self, logger)

    def _make_stem_layer(self) -> None:
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        self.conv1 = ConvModule(
            self.in_channels,
            self.base_channels,
            kernel_size=self.conv1_kernel,
            stride=(self.conv1_stride_t, self.conv1_stride_s, self.conv1_stride_s),
            padding=tuple([(k - 1) // 2 for k in _triple(self.conv1_kernel)]),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(self.pool1_stride_t, self.pool1_stride_s, self.pool1_stride_s),
            padding=(0, 1, 1),
        )

        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def _freeze_stages(self) -> None:
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    @staticmethod
    def _init_weights(self, pretrained: Optional[str] = None) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch.

        Args:
            pretrained (str | None): The path of the pretrained weight. Will
                override the original `pretrained` if set. The arg is added to
                be compatible with mmdet. Defaults to None.
        """
        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f"load model from: {self.pretrained}")

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights(logger)
            else:
                # Directly load 3D model.
                load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck3d):
                        constant_init(m.conv3.bn, 0)
                    elif isinstance(m, BasicBlock3d):
                        constant_init(m.conv2.bn, 0)
        else:
            raise TypeError("pretrained must be a str or None")

    def init_weights(self, pretrained: Optional[str] = None) -> None:
        """Initialize weights."""
        self._init_weights(self, pretrained)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor or tuple[torch.Tensor]: The feature of the input
            samples extracted by the backbone.
        """
        x = self.conv1(x)
        if self.with_pool1:
            x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i == 0 and self.with_pool2:
                x = self.pool2(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)

    def train(self, mode: bool = True) -> None:
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class ResNet3dPathway(ResNet3d):
    """A pathway of Slowfast based on ResNet3d.

    Args:
        lateral (bool): Determines whether to enable the lateral connection
            from another pathway. Defaults to False.
        lateral_inv (bool): Whether to use deconv to upscale the time
            dimension of features from another pathway. Defaults to False.
        lateral_norm (bool): Determines whether to enable the lateral norm
            in lateral layers. Defaults to False.
        speed_ratio (int): Speed ratio indicating the ratio between time
            dimension of the fast and slow pathway, corresponding to the
            ``alpha`` in the paper. Defaults to 8.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to ``beta`` in the paper.
            Defaults to 8.
        fusion_kernel (int): The kernel size of lateral fusion.
            Defaults to 5.
        lateral_infl (int): The ratio of the inflated channels.
            Defaults to 2.
        lateral_activate (list[int]): Flags for activating the lateral
            connection. Defaults to ``[1, 1, 1, 1]``.
    """

    def __init__(
        self,
        lateral: bool = False,
        lateral_inv: bool = False,
        lateral_norm: bool = False,
        speed_ratio: int = 8,
        channel_ratio: int = 8,
        fusion_kernel: int = 5,
        lateral_infl: int = 2,
        lateral_activate: List[int] = [1, 1, 1, 1],
        **kwargs,
    ) -> None:
        self.lateral = lateral
        self.lateral_inv = lateral_inv
        self.lateral_norm = lateral_norm
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio
        self.fusion_kernel = fusion_kernel
        self.lateral_infl = lateral_infl
        self.lateral_activate = lateral_activate
        self._calculate_lateral_inplanes(kwargs)

        super().__init__(**kwargs)
        self.inplanes = self.base_channels
        if self.lateral and self.lateral_activate[0] == 1:
            if self.lateral_inv:
                self.conv1_lateral = DeConvModule(
                    self.inplanes * self.channel_ratio,
                    self.inplanes * self.channel_ratio // lateral_infl,
                    kernel_size=(fusion_kernel, 1, 1),
                    stride=(self.speed_ratio, 1, 1),
                    padding=((fusion_kernel - 1) // 2, 0, 0),
                    with_bn=True,
                    with_relu=True,
                )
            else:
                self.conv1_lateral = ConvModule(
                    self.inplanes // self.channel_ratio,
                    self.inplanes * lateral_infl // self.channel_ratio,
                    kernel_size=(fusion_kernel, 1, 1),
                    stride=(self.speed_ratio, 1, 1),
                    padding=((fusion_kernel - 1) // 2, 0, 0),
                    bias=False,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg if self.lateral_norm else None,
                    act_cfg=self.act_cfg if self.lateral_norm else None,
                )

        self.lateral_connections = []
        for i in range(len(self.stage_blocks)):
            planes = self.base_channels * 2**i
            self.inplanes = planes * self.block.expansion

            if lateral and i != self.num_stages - 1 and self.lateral_activate[i + 1]:
                # no lateral connection needed in final stage
                lateral_name = f"layer{(i + 1)}_lateral"
                if self.lateral_inv:
                    conv_module = DeConvModule(
                        self.inplanes * self.channel_ratio,
                        self.inplanes * self.channel_ratio // lateral_infl,
                        kernel_size=(fusion_kernel, 1, 1),
                        stride=(self.speed_ratio, 1, 1),
                        padding=((fusion_kernel - 1) // 2, 0, 0),
                        bias=False,
                        with_bn=True,
                        with_relu=True,
                    )
                else:
                    conv_module = ConvModule(
                        self.inplanes // self.channel_ratio,
                        self.inplanes * lateral_infl // self.channel_ratio,
                        kernel_size=(fusion_kernel, 1, 1),
                        stride=(self.speed_ratio, 1, 1),
                        padding=((fusion_kernel - 1) // 2, 0, 0),
                        bias=False,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg if self.lateral_norm else None,
                        act_cfg=self.act_cfg if self.lateral_norm else None,
                    )
                setattr(self, lateral_name, conv_module)
                self.lateral_connections.append(lateral_name)

    def _calculate_lateral_inplanes(self, kwargs):
        """Calculate inplanes for lateral connection."""
        depth = kwargs.get("depth", 50)
        expansion = 1 if depth < 50 else 4
        base_channels = kwargs.get("base_channels", 64)
        lateral_inplanes = []
        for i in range(kwargs.get("num_stages", 4)):
            if expansion % 2 == 0:
                planes = base_channels * (2**i) * ((expansion // 2) ** (i > 0))
            else:
                planes = base_channels * (2**i) // (2 ** (i > 0))
            if self.lateral and self.lateral_activate[i]:
                if self.lateral_inv:
                    lateral_inplane = planes * self.channel_ratio // self.lateral_infl
                else:
                    lateral_inplane = planes * self.lateral_infl // self.channel_ratio
            else:
                lateral_inplane = 0
            lateral_inplanes.append(lateral_inplane)
        self.lateral_inplanes = lateral_inplanes

    def inflate_weights(self, logger: MMLogger) -> None:
        """Inflate the resnet2d parameters to resnet3d pathway.

        The differences between resnet3d and resnet2d mainly lie in an extra
        axis of conv kernel. To utilize the pretrained parameters in 2d model,
        the weight of conv2d models should be inflated to fit in the shapes of
        the 3d counterpart. For pathway the ``lateral_connection`` part should
        not be inflated from 2d weights.

        Args:
            logger (MMLogger): The logger used to print
                debugging information.
        """

        state_dict_r2d = _load_checkpoint(self.pretrained, map_location="cpu")
        if "state_dict" in state_dict_r2d:
            state_dict_r2d = state_dict_r2d["state_dict"]

        inflated_param_names = []
        for name, module in self.named_modules():
            if "lateral" in name:
                continue
            if isinstance(module, ConvModule):
                # we use a ConvModule to wrap conv+bn+relu layers, thus the
                # name mapping is needed
                if "downsample" in name:
                    # layer{X}.{Y}.downsample.conv->layer{X}.{Y}.downsample.0
                    original_conv_name = name + ".0"
                    # layer{X}.{Y}.downsample.bn->layer{X}.{Y}.downsample.1
                    original_bn_name = name + ".1"
                else:
                    # layer{X}.{Y}.conv{n}.conv->layer{X}.{Y}.conv{n}
                    original_conv_name = name
                    # layer{X}.{Y}.conv{n}.bn->layer{X}.{Y}.bn{n}
                    original_bn_name = name.replace("conv", "bn")
                if original_conv_name + ".weight" not in state_dict_r2d:
                    logger.warning(
                        f"Module not exist in the state_dict_r2d"
                        f": {original_conv_name}"
                    )
                else:
                    self._inflate_conv_params(
                        module.conv,
                        state_dict_r2d,
                        original_conv_name,
                        inflated_param_names,
                    )
                if original_bn_name + ".weight" not in state_dict_r2d:
                    logger.warning(
                        f"Module not exist in the state_dict_r2d"
                        f": {original_bn_name}"
                    )
                else:
                    self._inflate_bn_params(
                        module.bn,
                        state_dict_r2d,
                        original_bn_name,
                        inflated_param_names,
                    )

        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(state_dict_r2d.keys()) - set(inflated_param_names)
        if remaining_names:
            logger.info(
                f"These parameters in the 2d checkpoint are not loaded"
                f": {remaining_names}"
            )

    def _inflate_conv_params(
        self,
        conv3d: nn.Module,
        state_dict_2d: OrderedDict,
        module_name_2d: str,
        inflated_param_names: List[str],
    ) -> None:
        """Inflate a conv module from 2d to 3d.

        The differences of conv modules betweene 2d and 3d in Pathway
        mainly lie in the inplanes due to lateral connections. To fit the
        shapes of the lateral connection counterpart, it will expand
        parameters by concatting conv2d parameters and extra zero paddings.

        Args:
            conv3d (nn.Module): The destination conv3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding conv module in the
                2d model.
            inflated_param_names (list[str]): List of parameters that have been
                inflated.
        """
        weight_2d_name = module_name_2d + ".weight"
        conv2d_weight = state_dict_2d[weight_2d_name]
        old_shape = conv2d_weight.shape
        new_shape = conv3d.weight.data.shape
        kernel_t = new_shape[2]

        if new_shape[1] != old_shape[1]:
            if new_shape[1] < old_shape[1]:
                warnings.warn(
                    f"The parameter of {module_name_2d} is not"
                    "loaded due to incompatible shapes. "
                )
                return
            # Inplanes may be different due to lateral connections
            new_channels = new_shape[1] - old_shape[1]
            pad_shape = old_shape
            pad_shape = pad_shape[:1] + (new_channels,) + pad_shape[2:]
            # Expand parameters by concat extra channels
            conv2d_weight = torch.cat(
                (
                    conv2d_weight,
                    torch.zeros(pad_shape)
                    .type_as(conv2d_weight)
                    .to(conv2d_weight.device),
                ),
                dim=1,
            )

        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)

        if getattr(conv3d, "bias") is not None:
            bias_2d_name = module_name_2d + ".bias"
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)

    def _freeze_stages(self) -> None:
        """Prevent all the parameters from being optimized before
        `self.frozen_stages`."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            if i != len(self.res_layers) and self.lateral:
                # No fusion needed in the final stage
                lateral_name = self.lateral_connections[i - 1]
                conv_lateral = getattr(self, lateral_name)
                conv_lateral.eval()
                for param in conv_lateral.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained: Optional[str] = None) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if pretrained:
            self.pretrained = pretrained

        # Override the init_weights of i3d
        super().init_weights()
        for module_name in self.lateral_connections:
            layer = getattr(self, module_name)
            for m in layer.modules():
                if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                    kaiming_init(m)


pathway_cfg = {
    "resnet3d": ResNet3dPathway,
    # TODO: BNInceptionPathway
}


def build_pathway(cfg: Dict, *args, **kwargs) -> nn.Module:
    """Build pathway.

    Args:
        cfg (dict): cfg should contain:
            - type (str): identify backbone type.

    Returns:
        nn.Module: Created pathway.
    """
    if not (isinstance(cfg, dict) and "type" in cfg):
        raise TypeError('cfg must be a dict containing the key "type"')
    cfg_ = cfg.copy()

    pathway_type = cfg_.pop("type")
    if pathway_type not in pathway_cfg:
        raise KeyError(f"Unrecognized pathway type {pathway_type}")

    pathway_cls = pathway_cfg[pathway_type]
    pathway = pathway_cls(*args, **kwargs, **cfg_)

    return pathway


class ResNet3dSlowOnly(ResNet3dPathway):
    """SlowOnly backbone based on ResNet3dPathway.

    Args:
        conv1_kernel (Sequence[int]): Kernel size of the first conv layer.
            Defaults to ``(1, 7, 7)``.
        conv1_stride_t (int): Temporal stride of the first conv layer.
            Defaults to 1.
        pool1_stride_t (int): Temporal stride of the first pooling layer.
            Defaults to 1.
        inflate (Sequence[int]): Inflate dims of each block.
            Defaults to ``(0, 0, 1, 1)``.
        with_pool2 (bool): Whether to use pool2. Defaults to False.
    """

    def __init__(
        self,
        conv1_kernel: Sequence[int] = (1, 7, 7),
        conv1_stride_t: int = 1,
        pool1_stride_t: int = 1,
        inflate: Sequence[int] = (0, 0, 1, 1),
        with_pool2: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            conv1_kernel=conv1_kernel,
            conv1_stride_t=conv1_stride_t,
            pool1_stride_t=pool1_stride_t,
            inflate=inflate,
            with_pool2=with_pool2,
            **kwargs,
        )

        assert not self.lateral
