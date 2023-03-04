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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetA(nn.Module):
    def __init__(self, block, num_blocks, num_classes, in_planes=16,
                 pen_filters=64, pretrained=False):
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
        self.num_classes = num_classes
        self.apply(_weights_init)
        self._dropout = None

    def get_child(self, key):
        """Get child for corresponding :code:`key`

        Child is a :code:`torch.nn.conv.Conv2d` module.
        Key acceses the :code:`conv` module as [layer, block, conv]

        Indexing is according to default Resnet implementation in torchvision.
        Layers are named starting from 1
        Blocks are not named and retrieved from children so they start from 0.
        Conv modules are named from 1.

        (0, 0, 0) returns None and implies that intial Conv and BN modules should
        be returned.

        """
        if key == (0, 0, 0):
            return self.conv1, self.bn1
        elif key == (0, 0):
            return self.relu    # after self.bn1
        try:
            i, j, k = key
        except ValueError:
            i, j = key
        layer = getattr(self, f"layer{i}", None)
        try:
            block = layer and [*layer.children()][j]
        except Exception:
            block = None
        if len(key) == 2:
            return block
        else:
            conv = block and getattr(block, f"conv{k}", None)
            bn = block and getattr(block, f"bn{k}", None)
            return conv, bn

    def set_child(self, key, in_inds, out_inds, mask=False):
        if len(key) == 2:
            raise ValueError(f"Child cannot be set with a 2 len tuple {key}")
        if key == (0, 0, 0):
            conv, bn = self.conv1, self.bn1
        else:
            i, j, k = key
            layer = getattr(self, f"layer{i}", None)
            try:
                block = layer and [*layer.children()][j]
            except Exception:
                block = None
            conv = getattr(block, f"conv{k}")
            bn = getattr(block, f"bn{k}")
        if mask:
            in_inv = np.setdiff1d(np.arange(0, conv.weight.shape[1]), in_inds)
            out_inv = np.setdiff1d(np.arange(0, conv.weight.shape[0]), out_inds)
            conv.weight.data[:, in_inv, :, :] = 0
            conv.weight.data[out_inv] = 0
        else:
            if in_inds == -1:
                in_channels = conv.in_channels
            else:
                in_channels = len(in_inds)
            if out_inds == -1:
                out_channels = conv.out_channels
            else:
                out_channels = len(out_inds)
            conv_params = {"in_channels": in_channels,
                           "out_channels": out_channels,
                           "kernel_size": conv.kernel_size,
                           "stride": conv.stride,
                           "padding": conv.padding,
                           "bias": conv.bias}
            bn_params = {"num_features": out_channels,
                         "eps": bn.eps,
                         "momentum": bn.momentum,
                         "affine": bn.affine,
                         "track_running_stats": bn.track_running_stats}
            new_conv = nn.Conv2d(**conv_params)
            new_bn = nn.BatchNorm2d(**bn_params)
            # outgoing channel weights are indexed first
            if out_inds == -1:
                if in_inds == -1:
                    pass
                else:
                    new_conv.data = conv.weight[:, in_inds, :, :].clone()
            else:
                if in_inds == -1:
                    new_conv.data = conv.weight[out_inds].clone()
                else:
                    new_conv.data = conv.weight[out_inds][:, in_inds, :, :].clone
                new_bn.data = bn.weight[out_inds].clone()
            if key == (0, 0, 0):
                self.conv1 = new_conv
                self.bn1 = new_bn
            else:
                setattr(block, f"conv{k}", new_conv)
                setattr(block, f"bn{k}", new_bn)

    def compress(self, indices, mask=False):
        layers = []
        i = 1
        while getattr(self, f"layer{i}", None):
            layers.append(f"layer{i}")
            i += 1
        prev_inds = None
        for key, out_inds in indices.items():
            if key == (0, 0, 0):
                in_inds = [0, 1, 2]
            else:
                in_inds = prev_inds
            self.set_child(key, in_inds, out_inds, mask)
            prev_inds = out_inds

    def compress_at(self, indices, key):
        prev_inds = None
        for key, out_inds in indices.items():
            if key == (0, 0, 0):
                in_inds = [0, 1, 2]
            else:
                in_inds = prev_inds
            self.set_child(key, in_inds, out_inds)
            prev_inds = out_inds


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
