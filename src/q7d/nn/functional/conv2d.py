from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from torch import Tensor
import torch.nn.functional as F
import math

from ..defs import *


def conv2d(stride, padding, dilation, groups, quantizerBundle: PrefixBundle, accumulatorBundle: PrefixBundle) -> AutogradFunction:

    class _Conv2d(Function):
        @staticmethod
        def forward(ctx: FunctionCtx, input: Tensor, weight: Tensor, bias: Tensor):
            q_input = quantizerBundle.qx(input)
            q_weight = quantizerBundle.qw(weight)
            q_bias = None
            if bias is not None:
                q_bias = quantizerBundle.qb(bias)

            ctx.save_for_backward(q_input, q_weight, q_bias)
            output = F.conv2d(q_input, q_weight, q_bias,
                              stride, padding, dilation, groups)
            return quantizerBundle.qy(output)

        @staticmethod
        def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
            input: Tensor = ctx.saved_tensors[0]
            weight: Tensor = ctx.saved_tensors[1]
            bias: Tensor = ctx.saved_tensors[2]

            q_grad_output = quantizerBundle.qdy(grad_output)

            grad_input = None
            grad_weight = None
            grad_bias = None

            if ctx.needs_input_grad[0]:  # input
                hout_num = (input.shape[2] + 2 * padding[0] -
                        dilation[0] * (weight.shape[2] - 1) - 1)

                output_padding = 0

                floored =  math.floor(hout_num/stride[0]) * stride[0]

                output_padding = hout_num - floored

                grad_input = F.conv_transpose2d(q_grad_output, weight, stride=stride, padding=padding,
                                                dilation=dilation, groups=groups, output_padding=output_padding)

                grad_input = quantizerBundle.qdx(grad_input)

            if ctx.needs_input_grad[1]:  # weight
                # grad_output : (N, Cout, Hout, Wout)
                # grad_f : (N, Cout, L)
                grad_f = q_grad_output.view(
                    q_grad_output.shape[0], q_grad_output.shape[1], -1)

                # input_u : (N, L, Cin*kernel_size)
                input_u = F.unfold(
                    input, kernel_size=weight.shape[-2:], dilation=dilation, padding=padding, stride=stride).transpose(1, 2)

                # grads : (N, Cout, Cin*kernel_size)
                grads = grad_f @ input_u

                # grad_weight : (1, Cout, Cin, *kernel_size)
                grad_weight = accumulatorBundle.aw(grads).view(1, *weight.shape)

                grad_weight = quantizerBundle.qdw(grad_weight)

                # grad_weight : (Cout, Cin, *kernel_size)
                grad_weight = grad_weight.squeeze(0)

            if bias is not None and ctx.needs_input_grad[2]:  # bias
                # grad_output : (N, Cout, H, W)
                # grad_s : (N, Cout)
                grad_s = q_grad_output.sum((2, 3))

                # grad_bias : (1, Cout)
                grad_bias = accumulatorBundle.ab(grad_s)

                grad_bias = quantizerBundle.qdx(grad_bias)

                # grad_bias : (Cout)
                grad_bias = grad_bias.squeeze(0)

            return grad_input, grad_weight, grad_bias

    return _Conv2d.apply
