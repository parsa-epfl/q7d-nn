import torch
from torch import Tensor
import unittest
from q7d.nn.layers import QConv2d
from q7d.nn.defs import PrefixBundle
from q7d.quantizers import IntQuantizer
import torch.nn.functional as F
import torch.nn as nn


class Test(unittest.TestCase):

    def train(self, net, f, train_x, train_y):
        epoch = 30
        opt = torch.optim.SGD(net.parameters(), lr=0.1)
        criterion = nn.MSELoss()

        for i in range(epoch):

            for j in range(len(train_x)):
                y = net(train_x[j])

                loss = criterion(y, train_y[j])

                loss.backward()

                opt.step()

                net.zero_grad()

    def create_data(self, train_size, f):
        train_x = []
        train_y = []

        # Create data
        for i in range(train_size):
            x = torch.rand(1, 1, 14, 14)
            y = f(x)
            train_x.append(x)
            train_y.append(y)

        return train_x, train_y

    def test_baseline_training(self):
        w = torch.tensor([[[
            [4, -3.3, 1.2],
            [6, 0.6, 0.2],
            [0.9, -3.4, 1.14],
        ]]])

        b = torch.Tensor([3.6])

        def fabricated_function(x: Tensor) -> Tensor:
            return F.conv2d(x, w, b)

        train_size = 500
        train_x, train_y = self.create_data(train_size, fabricated_function)

        model = QConv2d(1, 1, 3)

        self.train(model, fabricated_function, train_x, train_y)

        torch.testing.assert_allclose(model.weight, w)
        torch.testing.assert_allclose(model.bias, b)

    def test_quantization_training(self):
        w = torch.tensor([[[
            [4, -3.3, 1.2],
            [6, 0.6, 0.2],
            [0.9, -3.4, 1.14],
        ]]])

        b = torch.Tensor([3.6])

        def fabricated_function(x: Tensor) -> Tensor:
            return F.conv2d(x, w, b)

        train_size = 1000
        train_x, train_y = self.create_data(train_size, fabricated_function)

        model = QConv2d(1, 1, 3, quantizerBundle=PrefixBundle(
            default=IntQuantizer(12)))

        self.train(model, fabricated_function, train_x, train_y)

        print(model.weight)
        torch.testing.assert_allclose(model.weight, w, atol=0.1, rtol=0.1)
        torch.testing.assert_allclose(model.bias, b, atol=0.1, rtol=0.1)


if __name__ == "__main__":
    unittest.main()
