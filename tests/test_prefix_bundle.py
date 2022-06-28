import torch
import unittest
from q7d.nn.defs import PrefixBundle

class Test(unittest.TestCase):
    def runTest(self) -> None:
        pb = PrefixBundle()
        pb.qdx = lambda x: x
        pb.qd = lambda x: x + 3
        pb.q = lambda x: x +  5

        torch.testing.assert_allclose(pb.qdx(2), 2)
        torch.testing.assert_allclose(pb.q(7), 12)
        torch.testing.assert_allclose(pb.qy(8), 13)
        torch.testing.assert_allclose(pb.qdy(8), 11)

if __name__ == "__main__":
    unittest.main()
