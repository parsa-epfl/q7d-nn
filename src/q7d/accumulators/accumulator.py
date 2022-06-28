from torch import Tensor
from .hsum import *


class GradientAccumulator:
    """
    Gradient accumulator class, accumulates the gradients for a parameter of a layer. The Tensor given as input must have at least 2 dimensions. 
    And the accumulation is done along the first dimension.
    """

    def __init__(self) -> None:
        pass

    def accumulate(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, t: Tensor) -> Tensor:
        return self.accumulate(t)


class DefaultAccumulator(GradientAccumulator):
    """
    Default accumulator, just a sum.
    """

    def __init__(self) -> None:
        super().__init__()

    def accumulate(self, t: Tensor) -> Tensor:
        return t.sum(0, keepdim=True)


class HsumAccumulator(GradientAccumulator):
    """
    Performs hierarchical sum on the first dimension of the tensor given as input, applying quantization
    """

    def __init__(self, reducer: HierarchicalReducer) -> None:
        super().__init__()
        self._reducer = reducer

    def accumulate(self, t: Tensor) -> Tensor:
        return hsum(t, self._reducer)
