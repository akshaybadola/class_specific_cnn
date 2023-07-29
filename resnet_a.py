import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


__all__ = ['resnet18', 'resnet20', 'resnet32', 'resnet34', 'resnet44',
           'resnet56', 'resnet110', 'resnet1202']


from resnet_common import _weights_init, is_tracked, LambdaLayer


class BasicBlockA(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', **kwargs):
        super(BasicBlockA, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNetA paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2],
                                                  (0, 0, 0, 0, planes//4, planes//4),
                                                  "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class ResNetA(nn.Module):
    def __init__(self, block, num_blocks, num_classes, in_planes=16,
                 pen_filters=64, indices=None, pretrained=False):
        super(ResNetA, self).__init__()
        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(pen_filters, num_classes)
        self.indices = indices
        self.num_classes = num_classes
        self.apply(_weights_init)
        # Decompose weights. It's a view of self.linear
        if self.indices is not None and len(self.indices):
            self.final_weights = nn.Parameter(torch.stack([self.linear.weight.T[indices[i].tolist(), i]
                                                           for i in range(len(indices))]))
            self.final_bias = nn.Parameter(self.linear.bias)

    def _init_final_weights(self):
        self.final_weights.data.copy_(torch.stack(
            [self.linear.weight.data.T[self.indices[i], i]
             for i in range(len(self.indices))]))
        self.final_bias.data = self.linear.bias.data.clone()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def head(self, x):                          # Input (128, 3, 32, 32)
        out = self.relu(self.bn1(self.conv1(x)))   # (128, 16, 32, 32)
        out = self.layer1(out)                  # (128, 16, 32, 32)
        out = self.layer2(out)                  # (128, 32, 16, 16)
        out = self.layer3(out)                  # (128, 64, 8, 8)
        return out

    def tail(self, x):                          # (128, 32, 16, 16)
        out = F.avg_pool2d(x, x.size()[3])      # (128, 64, 1, 1)
        out = out.view(out.size(0), -1)         # (128, 64)
        return self.linear(out)                  # (128, 10)

    def forward(self, x):
        return self.tail(self.head(x))

    def forward_decomposed(self, x):
        x = self.head(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = x[:, self.indices]
        return ((x * self.final_weights).sum(-1) + self.final_bias)

    def forward_noise_at_inds_for_label(self, x):
        val_label = getattr(self, "_val_label", None)
        inds = self.indices[-1][val_label]
        invert_inds = set(range(64)) - set(inds)
        invert_inds = np.array([*invert_inds])
        if val_label is None:
            raise AttributeError("Model has no attribute _val_label")
        x = self.head(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x_1 = x.clone()
        x_2 = x.clone()
        x_1[:, inds] = torch.randn(x_1.shape[0], len(inds)).to(x_1.device)
        x_2[:, invert_inds] = torch.randn(x_2.shape[0], len(invert_inds)).to(x_2.device)
        return self.linear(x_1), self.linear(x_2)


def resnet20(num_classes, pretrained=None, **kwargs):
    if pretrained:
        print("Warning! This model does not support \"pretrained\".")
    return ResNetA(BasicBlockA, [3, 3, 3], num_classes=num_classes, **kwargs)


def resnet32(num_classes, pretrained=None, **kwargs):
    if pretrained:
        print("Warning! This model does not support \"pretrained\".")
    return ResNetA(BasicBlockA, [5, 5, 5], num_classes=num_classes, **kwargs)


def resnet44(num_classes, pretrained=None, **kwargs):
    if pretrained:
        print("Warning! This model does not support \"pretrained\".")
    return ResNetA(BasicBlockA, [7, 7, 7], num_classes=num_classes, **kwargs)


def resnet56(num_classes, pretrained=None, **kwargs):
    if pretrained:
        print("Warning! This model does not support \"pretrained\".")
    return ResNetA(BasicBlockA, [9, 9, 9], num_classes=num_classes, **kwargs)


def resnet110(num_classes, pretrained=None, **kwargs):
    if pretrained:
        print("Warning! This model does not support \"pretrained\".")
    return ResNetA(BasicBlockA, [18, 18, 18], num_classes=num_classes)


def resnet1202(num_classes, pretrained=None, **kwargs):
    if pretrained:
        print("Warning! This model does not support \"pretrained\".")
    return ResNetA(BasicBlockA, [200, 200, 200], num_classes=num_classes, **kwargs)
