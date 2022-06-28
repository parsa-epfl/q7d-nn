import torch.nn as nn
from torch import Tensor
from ..quantizers import IdQuantizer
from ..accumulators import DefaultAccumulator
from .functional import *
from .defs import *


class QLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 quantizerBundle=PrefixBundle(default=IdQuantizer()),
                 accumulatorBundle=PrefixBundle(default=DefaultAccumulator())
                 ) -> None:
        super().__init__(in_features, out_features, bias)
        self.linear_function = linear(quantizerBundle, accumulatorBundle)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_function(x, self.weight, self.bias)


class QConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 quantizerBundle=PrefixBundle(default=IdQuantizer()),
                 accumulatorBundle=PrefixBundle(default=DefaultAccumulator())
                 ) -> None:
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias)
        self.conv_function = conv2d(self.stride, self.padding, self.dilation, self.groups, quantizerBundle, accumulatorBundle)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_function(x, self.weight, self.bias)
