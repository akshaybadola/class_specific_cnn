'''
Adapted from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from common_pyutil.monitor import Timer


from resnet_common import (get_order, make_grid, inds_at_t, conv_with_w_at_t,
                           _weights_init, conv_with_w_for_inds, is_tracked)


__all__ = ["resnet50", "resnet101"]


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


timer = Timer()


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockB(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockB, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlockB only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlockB")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # import pdb; pdb.set_trace()
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, **kwargs):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if "num_layer" in kwargs:
            self._num_layer = kwargs["num_layer"]
            self._num_block = kwargs["num_block"]
        if "num_layer" in kwargs and kwargs["indices"]:
            self.indices = kwargs["indices"]
            keys = [(self._num_layer, self._num_block, 1),
                    (self._num_layer, self._num_block, 2),
                    (self._num_layer, self._num_block, 3)]
            for key in keys:
                if key in self.indices.keys and len(self.indices[key]):
                    inds = self.indices[key]
                    if key[-1] == 1:
                        self.convlist1 = nn.ModuleList(
                            [nn.Conv2d(inplanes, len(i),
                                       kernel_size=self.conv1.kernel_size,
                                       stride=1, bias=False)
                             for i in inds])
                        self.bnlist1 = nn.ModuleList([nn.BatchNorm2d(len(i))
                                                      for i in inds])
                    if key[-1] == 2:
                        self.convlist2 = nn.ModuleList(
                            [nn.Conv2d(planes, len(i),
                                       kernel_size=self.conv2.kernel_size,
                                       stride=1, padding=1, bias=False)
                             for i in inds])
                        self.bnlist2 = nn.ModuleList([nn.BatchNorm2d(len(i))
                                                      for i in inds])
                    if key[-1] == 3:
                        self.convlist3 = nn.ModuleList(
                            [nn.Conv2d(planes, len(i),
                                       kernel_size=self.conv3.kernel_size,
                                       stride=1, bias=False)
                             for i in inds])
                        self.bnlist3 = nn.ModuleList([nn.BatchNorm2d(len(i))
                                                      for i in inds])
        else:
            self.indices = None
        self.save_inds = False
        self.track_stats = False

    def forward(self, x):
        # if self._num_layer == 4:
        #     import pdb; pdb.set_trace()
        identity = x

        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)

        # out = self.conv3(out)
        # out = self.bn3(out)

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        # out += identity
        # out = self.relu(out)

        # NOTE: We either track or decompose. We should be able to do both
        out_list = []
        # Block 1
        if self.track_stats and\
           is_tracked(self.track_for, self._num_layer, self._num_block, 1):
            out = F.relu(self.bn1(self.conv1(x)))
            key = (self._num_layer, self._num_block, 1)
            ret = inds_at_t(out.detach().cpu().numpy())
            inds, sum, csum = np.int32(ret["inds"]), ret["sum"], ret["cumsum"]
            for i, label in enumerate(self.labels):
                self.order_stats[label][key].extend([("inds", inds[i]),
                                                     ("sum", sum[i]),
                                                     ("csum", csum[i])])
        elif hasattr(self, "_num_layer") and hasattr(self, "indices") and\
             self.indices and (self._num_layer, self._num_block, 1) in self.indices.keys:
            inds = self.indices[self._num_layer, self._num_block, 1]
            for c, b, i in zip(self.convlist1, self.bnlist1, inds):
                out_list.append(F.relu(b(c(x))))
        else:
            out = F.relu(self.bn1(self.conv1(x)))

        # Block 2
        if self.track_stats and\
           is_tracked(self.track_for, self._num_layer, self._num_block, 2):
            out = F.relu(self.bn2(self.conv2(out)))
            key = (self._num_layer, self._num_block, 2)
            ret = inds_at_t(out.detach().cpu().numpy())
            inds, sum, csum = np.int32(ret["inds"]), ret["sum"], ret["cumsum"]
            for i, label in enumerate(self.labels):
                self.order_stats[label][key].extend([("inds", inds[i]),
                                                     ("sum", sum[i]),
                                                     ("csum", csum[i])])
        elif hasattr(self, "_num_layer") and hasattr(self, "indices") and\
             self.indices and (self._num_layer, self._num_block, 2) in self.indices.keys:
            inds = self.indices[self._num_layer, self._num_block, 2]
            if out_list:
                for j, (c, b, i) in enumerate(zip(self.convlist2, self.bnlist2, inds)):
                    out_list[j] = F.relu(b(c(out_list[j])) + x[:, i, :, :])
            else:
                for c, b, i in zip(self.convlist2, self.bnlist2, inds):
                    out_list.append(F.relu(b(c(out)) + x[:, i, :, :]))
            return out_list
        else:
            out = self.bn2(self.conv2(out))

        # # Earlier Block 2 code
        # if hasattr(self, "indices") and\
        #    (self._num_layer, self._num_block, 2) in self.indices.keys:
        #     inds = self.indices[self._num_layer, self._num_block, 2]
        #     test = []
        #     for c, b, i in zip(self.convlist2, self.bnlist2, inds):
        #         test.append(F.relu(b(c(out)) + x[:, i, :, :]))
        #     return test
        # else:
        #     out = self.relu(self.bn2(self.conv2(out)))

        # # Earlier Block 3 code
        # if self.downsample is not None:
        #     identity = self.downsample(x)
        # if hasattr(self, "indices") and\
        #    (self._num_layer, self._num_block, 3) in self.indices.keys:
        #     inds = self.indices[self._num_layer, self._num_block, 3]
        #     test = []
        #     for c, b, i in zip(self.convlist3, self.bnlist3, inds):
        #         test.append(F.relu(b(c(out)) + x[:, i, :, :]))
        #     return test
        # else:
        #     out = self.relu(self.bn3(self.conv3(out)) + identity)

        # Block 3
        if self.downsample is not None:
            identity = self.downsample(x)
        if self.track_stats and\
           is_tracked(self.track_for, self._num_layer, self._num_block, 3):
            out = self.relu(self.bn3(self.conv3(out)) + identity)
            key = (self._num_layer, self._num_block, 3)
            ret = inds_at_t(out.detach().cpu().numpy())
            inds, sum, csum = np.int32(ret["inds"]), ret["sum"], ret["cumsum"]
            for i, label in enumerate(self.labels):
                self.order_stats[label][key].extend([("inds", inds[i]),
                                                     ("sum", sum[i]),
                                                     ("csum", csum[i])])
            return out
        elif hasattr(self, "_num_layer") and hasattr(self, "indices") and\
             self.indices and (self._num_layer, self._num_block, 3) in self.indices.keys:
            inds = self.indices[self._num_layer, self._num_block, 3]
            if out_list:
                for j, (c, b, i) in enumerate(zip(self.convlist3, self.bnlist3, inds)):
                    out_list[j] = F.relu(b(c(out_list[j])) + x[:, i, :, :])
            else:
                for c, b, i in zip(self.convlist3, self.bnlist3, inds):
                    out_list.append(F.relu(b(c(out)) + x[:, i, :, :]))
            return out_list
        else:
            out = self.relu(self.bn3(self.conv3(out)) + identity)
            return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, pretrained=False, indices=[]):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # These may be set later
        self.track_stats = False
        self.save_inds = False
        self.indices = indices

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_layer=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       num_layer=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       num_layer=3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       num_layer=4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockB):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, num_layer=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            num_layer=num_layer, num_block=0,
                            indices=self.indices))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,
                                num_layer=num_layer, num_block=i,
                                indices=self.indices))

        return nn.Sequential(*layers)

    # def _make_layer(self, block, planes, blocks, stride=1, dilate=False, num_layer=None):
    #     norm_layer = self._norm_layer
    #     downsample = None
    #     previous_dilation = self.dilation
    #     if dilate:
    #         self.dilation *= stride
    #         stride = 1
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             conv1x1(self.inplanes, planes * block.expansion, stride),
    #             norm_layer(planes * block.expansion),
    #         )

    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
    #                         self.base_width, previous_dilation, norm_layer))
    #     self.inplanes = planes * block.expansion
    #     for _ in range(1, blocks):
    #         layers.append(block(self.inplanes, planes, groups=self.groups,
    #                             base_width=self.base_width, dilation=self.dilation,
    #                             norm_layer=norm_layer))

    #     return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.track_stats:
            key = (0, 0, 0)
            if key in self.track_for:
                order = get_order(x)
                for k, v in order.items():
                    for i, label in enumerate(self.labels):
                        self.order_stats[label][key].append([k, v[i]])
        x = self.maxpool(x)
        if self.track_stats:
            key = (0, 0, 1)
            if key in self.track_for:
                order = get_order(x)
                for k, v in order.items():
                    for i, label in enumerate(self.labels):
                        self.order_stats[label][key].append([k, v[i]])
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNetB(torch.nn.Module):
    def __init__(self, arch, num_classes, pretrained=False, new_final_layer=False):
        super().__init__()
        self._resnet = None
        self.num_classes = num_classes
        if arch == 34:
            self._resnet = _resnet34(pretrained=pretrained)
        elif arch == 18:
            self._resnet = _resnet18(pretrained=pretrained)
        elif arch == 50:
            self._resnet = _resnet50(pretrained=pretrained)
        elif arch == 101:
            self._resnet = _resnet101(pretrained=pretrained)
        self.head_layers = nn.Sequential(*list(self._resnet.children())[:-1])
        self.conv1 = self._resnet.conv1
        self.bn1 = self._resnet.bn1
        self.relu = self._resnet.relu
        self.maxpool = self._resnet.maxpool
        self.layer1 = self._resnet.layer1
        self.layer2 = self._resnet.layer2
        self.layer3 = self._resnet.layer3
        self.layer4 = self._resnet.layer4
        self.avgpool = self._resnet.avgpool
        self.fc = self._resnet.fc
        # i = 0
        # for child in self.head_layers.children():
        #     if isinstance(child, nn.Sequential):
        #         for _child in child:
        #             if isinstance(_child, Bottleneck) or isinstance(_child, BasicBlockA)\
        #                or isinstance(_child, BasicBlockB):
        #                 setattr(self, f"child_{i:03d}", child)
        # self.num_children = i + 1
        model_num_classes, pen_filters = [*self._resnet.children()][-1].weight.shape
        self.pen_filters = pen_filters
        if pretrained:
            if num_classes != model_num_classes and not new_final_layer:
                raise ValueError(f"The number of classes {model_num_classes} " +
                                 f"in pretrained model different than {num_classes}")
            if new_final_layer:
                print(f"Creating new final layer with {num_classes} classes")
                self.fc = nn.Linear(pen_filters, num_classes)
            else:
                self.fc = [*self._resnet.children()][-1]
        else:
            self.fc = nn.Linear(pen_filters, num_classes)
        self.csg = nn.Parameter(torch.full(
            (num_classes, pen_filters), 0.5, requires_grad=True))
        self._dropout = None
        self.forward_std = self.forward_std_b

    def head(self, x):
        return self.head_layers(x)

    def tail(self, x):
        if self._dropout:
            return self.fc(F.dropout(torch.flatten(x, start_dim=1), self._dropout))
        else:
            return self.fc(torch.flatten(x, start_dim=1))

    def forward(self, x, path, k=1, target_var=None):
        if path == path_dict['CSG']:
            return self.forward_csg(x)
        elif path == path_dict['VAL']:
            return self.forward_csg_val_alt(x, target_var, k)
        else:
            return self.forward_std(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        if getattr(torchvision.models, "util", None):
            func = torchvision.models.util.load_state_dict_from_url
        elif hasattr(torchvision, "_internally_replaced_utils"):
            func = torchvision._internally_replaced_utils.load_state_dict_from_url
        else:
            try:
                from torch.hub import load_state_dict_from_url
            except ImportError:
                from torch.utils.model_zoo import load_url as load_state_dict_from_url
            func = load_state_dict_from_url
        print(f"Loading pretrained model {arch}")
        state_dict = func(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(num_classes, pretrained=False, **kwargs):
    # return ResNet(18, num_classes=num_classes, pretrained=pretrained)
    return _resnet18(pretrained=pretrained, num_classes=num_classes, **kwargs)


def resnet34(num_classes, pretrained=False, **kwargs):
    # return ResNet(34, num_classes=num_classes, pretrained=pretrained)
    return _resnet34(pretrained=pretrained, num_classes=num_classes, **kwargs)


def resnet50(num_classes, pretrained=False, **kwargs):
    # return ResNet(50, num_classes=num_classes, pretrained=pretrained)
    return _resnet50(pretrained=pretrained, num_classes=num_classes, **kwargs)


def resnet101(num_classes, pretrained=False, **kwargs):
    # return ResNet(101, num_classes=num_classes, pretrained=pretrained)
    return _resnet101(pretrained=pretrained, num_classes=num_classes, **kwargs)


def resnet152(num_classes, pretrained=None, **kwargs):
    # return ResNet(152, num_classes=num_classes, pretrained=pretrained)
    return _resnet152(pretrained=pretrained, num_classes=num_classes, **kwargs)


def _resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlockB, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def _resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlockB, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def _resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def _resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def _resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


