import torch
from torch import Tensor
import math
from typing import Tuple, List


class Quantizer:
    """
    Quantizers class, can perform quantization, dequantization or both if you just want to introduce quantization error.
    """

    def __init__(self) -> None:
        pass

    def quantize(self, t: Tensor) -> Tuple[Tensor, object]:
        raise NotImplementedError

    def dequantize(self, t: Tensor, ctx: object) -> Tensor:
        raise NotImplementedError

    def fake_quantization(self, t: Tensor) -> Tensor:
        return self.dequantize(*self.quantize(t))

    def __call__(self, t: Tensor) -> Tensor:
        return self.fake_quantization(t)


class TiledQuantizer(Quantizer):

    def __init__(self, dims: List[int], q: Quantizer) -> None:
        super().__init__()
        self._dims = dims
        self._q = q

    def quantize(self, t: Tensor) -> Tuple[Tensor, object]:
        original_shape = t.size()
        tt = t.view(-1, *self._dims, t.size()[2:])

        result, ctx = self._q.quantize(tt)

        return result.view(original_shape), ctx

    def dequantize(self, t: Tensor, ctx: object) -> Tensor:
        original_shape = t.size()
        tt = t.view(-1, *self._dims, t.size()[2:])

        result = self._q.dequantize(tt)

        return result.view(original_shape)


class IntQuantizer(Quantizer):
    """
    Maps FP32 to integers, the bitwidth is given as parameter.
    """

    def __init__(self, m: int) -> None:
        super().__init__()
        self._m = m

    def quantize(self, t: Tensor) -> Tuple[Tensor, object]:
        shape = t.size()
        t_f = t.view(shape[0], -1)

        tmin = torch.min(t_f, dim=1)[0]
        tmax = torch.max(t_f, dim=1)[0]

        for i in ((tmax - tmin) == 0).nonzero():
            index = i.item()
            tmin[index] = 0
            tmax[index] = 1

        tmin_s = tmin.view(-1, 1)
        tmax_s = tmax.view(-1, 1)

        result = (t_f - tmin_s) / (tmax_s - tmin_s) * ((1 << self._m) - 1)
        return result.round().view(shape), (tmin_s, tmax_s)

    def dequantize(self, t: Tensor, ctx: object) -> Tensor:
        tmin, tmax = ctx

        shape = t.size()
        t_f = t.view(shape[0], -1)

        result = t_f / ((1 << self._m) - 1) * (tmax - tmin) + tmin

        return result.view(shape)


class FloatingPointQuantizer(Quantizer):
    """
    Just gets rid of some mantissa bits, exponent is not changed.
    """

    def __init__(self, m: int) -> None:
        super().__init__()
        self._m = m
        self._2m = math.pow(2.0, m)
        self._2mm = math.pow(2.0, -m)

    def quantize(self, t: Tensor) -> Tuple[Tensor, object]:
        m, e = t.frexp()
        m = m * 2  # fp convention
        e = e - 1  # re align
        sign = m.sign()
        m = sign * m
        q: Tensor = (m * self._2m).floor() * self._2mm
        return (sign * q * torch.pow(2.0, e), None)

    def dequantize(self, t: Tensor, ctx: object) -> Tensor:
        return t


class IdQuantizer(Quantizer):
    """
    Returns the tensor given as input, without changing anything.
    """

    def __init__(self) -> None:
        super().__init__()

    def quantize(self, t: Tensor) -> Tuple[Tensor, object]:
        return (t, None)

    def dequantize(self, t: Tensor, ctx: object) -> Tensor:
        return t
