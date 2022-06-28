from black import err
import torch
import unittest
from q7d.quantizers import IntQuantizer

class Test(unittest.TestCase):
    def test_int_quantizer(self) -> None:
        bits = 8
        q = IntQuantizer(bits)
        t = torch.rand(5, 5)
        qt = q(t)

        min = torch.min(t)
        max = torch.max(t)
        error = 1 /( (max - min) * ((1 << bits) - 1))
        torch.testing.assert_allclose(t, qt, atol=error, rtol=error)

if __name__ == "__main__":
    unittest.main()
