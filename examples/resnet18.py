import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
from q7d.nn.layers import QLinear, QConv2d
from torch.nn import Conv2d
from q7d.nn.defs import PrefixBundle
from q7d.quantizers import IntQuantizer, IdQuantizer, FloatingPointQuantizer
from q7d.accumulators import DefaultAccumulator, HsumAccumulator


def prepare_data(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ntr = len(trainset)
    nte = len(testset)
    print('Loaded {} tr and {} va samples.'.format(ntr, nte))

    return trainset, trainloader, testset, testloader, classes


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, quantizerBundle, accumulatorBundle, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = QConv2d(in_planes, planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, quantizerBundle=quantizerBundle, accumulatorBundle=accumulatorBundle)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=1,
                             padding=1, bias=False, quantizerBundle=quantizerBundle, accumulatorBundle=accumulatorBundle)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                QConv2d(in_planes, self.expansion*planes, kernel_size=1,
                        stride=stride, bias=False, quantizerBundle=quantizerBundle, accumulatorBundle=accumulatorBundle),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, quantizerBundle, accumulatorBundle, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = QConv2d(3, 64, kernel_size=3, stride=1,
                             padding=1, bias=False, quantizerBundle=quantizerBundle, accumulatorBundle=accumulatorBundle)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(
            block, 64, num_blocks[0], quantizerBundle, accumulatorBundle, stride=1)
        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], quantizerBundle, accumulatorBundle, stride=2)
        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], quantizerBundle, accumulatorBundle, stride=2)
        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], quantizerBundle, accumulatorBundle, stride=2)
        self.linear = QLinear(512*block.expansion,
                              num_classes, quantizerBundle=quantizerBundle, accumulatorBundle=accumulatorBundle)

    def _make_layer(self, block, planes, num_blocks, quantizerBundle, accumulatorBundle, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes,
                          quantizerBundle, accumulatorBundle, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(quantizerBundle: PrefixBundle, accumulatorBundle: PrefixBundle):
    return ResNet(BasicBlock, [2, 2, 2, 2], quantizerBundle, accumulatorBundle)


def train(net, trainloader, testloader, epochs, lr):
    device = torch.device(('cpu', 'cuda')[torch.cuda.is_available()])
    print(torch.cuda.get_device_name(device), flush=True)

    print(device, flush=True)

    net.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    for epoch in range(epochs):
        net.train()
        for it, batch in enumerate(trainloader):
            inputs, labels = [d.to(device) for d in batch]
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            #print('[{}, {}] loss: {:.3f}'.format(epoch + 1, it + 1, loss.item()), end="\r")

        print(epoch + 1, flush=True)
        test_model(net, testloader)

    print('Finished Training', flush=True)

    return net


def test_model(net, testloader):
    device = torch.device(('cpu', 'cuda')[torch.cuda.is_available()])

    # How the network performs on the whole dataset.
    correct = 0
    total = 0
    with torch.no_grad():
        for it, batch in enumerate(testloader):
            images, labels = [d.to(device) for d in batch]
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('The accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total), flush=True)


def resnet18_cifar10(args):
    trainset, trainloader, testset, testloader, classes = prepare_data(
        args.batch_size)

    if args.quant == 'int':
        qb = PrefixBundle(default=IntQuantizer(args.nb_bits))
    elif args.quant == 'fp':
        qb = PrefixBundle(default=FloatingPointQuantizer(args.nb_bits))
    else:
        qb = PrefixBundle(default=IdQuantizer())

    if args.acc == 'int':
        h = [(64, IdQuantizer()), (8, IntQuantizer(args.nb_bits))]
    elif args.acc == 'fp':
        h = [(64, IdQuantizer()), (8, FloatingPointQuantizer(args.nb_bits))]
    else:
        h = [(64, IdQuantizer()), (8, IdQuantizer())]
    
    ab = PrefixBundle(default=HsumAccumulator(h))

    net = ResNet18(qb, ab)
    net = train(net, trainloader, testloader, args.epoch, args.lr)
    if(args.lr2 is not None):
        print("Changing learning rate...", flush=True)
        net = train(net, trainloader, testloader, args.epoch2, args.lr2)
    test_model(net, testloader)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--lr2', type=float, required=False)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--epoch2', type=int, required=False)
    parser.add_argument('--quant', default='none', choices=['int', 'fp'])
    parser.add_argument('--acc', default='none', choices=['int', 'fp'])
    parser.add_argument('--nb_bits', type=int, required=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    resnet18_cifar10(args)
