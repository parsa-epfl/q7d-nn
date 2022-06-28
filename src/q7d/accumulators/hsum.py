from functools import reduce
import torch
import math
from typing import Callable, List, Tuple

TensorTransformer = Callable[[torch.Tensor], torch.Tensor]
HierarchicalReducer = List[Tuple[int, TensorTransformer]]


def hsum(t: torch.Tensor, reducer: HierarchicalReducer) -> torch.Tensor:
    """
    Performs hierachical sum on the first dimension of the tensor. Quantization can be added during summation.
    """

    assert len(t.shape) >= 1, "Not enough dimensions"

    def apply(t: torch.Tensor, n: int, transform: TensorTransformer) -> torch.Tensor:
        tt = transform(t)

        dnew = math.ceil(tt.shape[0] / n)
        result = torch.zeros(dnew, *tt.shape[1:]).to(t.device)

        for i in range(dnew):
            begin = i * n
            end = (i + 1) * n
            result[i] = torch.sum(tt[begin:end], dim=0, keepdim=True)

        return result

    for (n, transform) in reducer:
        t = apply(t, n, transform)

    return t
