from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from torch import Tensor
import torch.nn.functional as F

from ..defs import *


def _ensure_shape(t: Tensor) -> Tensor:
    if len(t.shape) < 3:
        return t.view(t.shape[0], 1, *t.shape[1:])
    return t


def linear(quantizerBundle: PrefixBundle, accumulatorBundle: PrefixBundle) -> AutogradFunction:

    class _Linear(Function):
        @staticmethod
        def forward(ctx: FunctionCtx, input: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
            q_input = quantizerBundle.qx(input)
            q_weight = quantizerBundle.qw(weight)
            q_bias = None
            if bias is not None:
                q_bias = quantizerBundle.qb(bias)

            ctx.save_for_backward(q_input, q_weight, q_bias)
            output = F.linear(q_input, q_weight, q_bias)
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
                grad_input = q_grad_output @ weight
                grad_input = quantizerBundle.qdx(grad_input)

            # q_grad_output : (N, *, Hout)
            q_grad_output = _ensure_shape(q_grad_output)

            # input : (N, *, Hin)
            input = _ensure_shape(input)

            if ctx.needs_input_grad[1]:  # weight
                # g_f : (N, D, Hout)
                g_f = q_grad_output.flatten(1, -2)

                # x_f : (N, D, Hin)
                input_f = input.flatten(1, -2)

                # g_f_p : (N, Hout, D)
                g_f_p = g_f.transpose(1, 2)

                # grads : (N, Hout, Hin) -- summed for the batch dimension
                grads = g_f_p @ input_f

                # grad_weight (1, Hout, Hin)
                grad_weight = accumulatorBundle.aw(grads)

                grad_weight = quantizerBundle.qdw(grad_weight)

                # grad_weight (Hout, Hin)
                grad_weight = grad_weight.squeeze(0)


            if bias is not None and ctx.needs_input_grad[2]:  # bias
                # g_f : (N, D, Hout) -- summed for the batch dimension
                g_f = q_grad_output.flatten(1, -2)

                # acc : (1, D, Hout)
                acc = accumulatorBundle.ab(g_f)

                # grad_bias (1, Hout)
                grad_bias = acc.sum(1)

                grad_bias = quantizerBundle.qdb(grad_bias)

                # grad_bias (Hout)
                grad_bias = grad_bias.squeeze(0)

            return grad_input, grad_weight, grad_bias

    return _Linear.apply
