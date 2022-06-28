import torch
import unittest
from q7d.accumulators import hsum

class Test(unittest.TestCase):
    def runTest(self) -> None:
        t = torch.tensor([
            [2, 3, 4, 5, 6],
            [10, 11, 12, 13, 14],
            [2, 3, 4, 5, 6],
            [10, 11, 12, 13, 14],
        ])

        def q1(x: torch.Tensor) -> torch.Tensor:
            return x.apply_(lambda x: x if x >= 10 else 0)

        def q2(x: torch.Tensor) -> torch.Tensor:
            return x.apply_(lambda x: x if x >= 12 else 0)

        result = hsum(t, [(2, q1), (2, q2)])
        expected = torch.tensor([[ 0.,  0., 24., 26., 28.]])

        torch.testing.assert_allclose(result, expected)

if __name__ == "__main__":
    unittest.main()
