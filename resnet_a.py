import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


__all__ = ['resnet18', 'resnet20', 'resnet32', 'resnet34', 'resnet44',
           'resnet56', 'resnet110', 'resnet1202']


from resnet_common import (get_order, make_grid, inds_at_t, conv_with_w_at_t,
                           _weights_init, conv_with_w_for_inds, is_tracked,
                           LambdaLayer)


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
        if "num_layer" in kwargs:
            self._num_layer = kwargs["num_layer"]
            self._num_block = kwargs["num_block"]
            self.indices = kwargs["indices"]
            keys = [(self._num_layer, self._num_block, 1),
                    (self._num_layer, self._num_block, 2)]
            for k, key in enumerate(keys):
                if key in self.indices.keys and len(self.indices[key]):
                    inds = self.indices[key]
                    if key[-1] == 1:
                        self.convlist1 = nn.ModuleList(
                            [nn.Conv2d(in_planes, len(i), kernel_size=3,
                                       stride=1, padding=1, bias=False)
                             for i in inds])
                        self.bnlist1 = nn.ModuleList([nn.BatchNorm2d(len(i))
                                                      for i in inds])
                    if key[-1] == 2:
                        # If previous layer is also decomposed
                        if keys[k-1] in self.indices.keys and len(self.indices[key]):
                            inds_a = self.indices[keys[k-1]]
                            inds_b = self.indices[key]
                            self.convlist2 = nn.ModuleList(
                                [nn.Conv2d(len(i), len(j), kernel_size=3,
                                           stride=1, padding=1, bias=False)
                                 for i, j in zip(inds_a, inds_b)])
                            self.bnlist2 = nn.ModuleList([nn.BatchNorm2d(len(j))
                                                          for j in inds_b])
                        else:
                            self.convlist2 = nn.ModuleList(
                                [nn.Conv2d(planes, len(i), kernel_size=3,
                                           stride=1, padding=1, bias=False)
                                 for i in inds])
                            self.bnlist2 = nn.ModuleList([nn.BatchNorm2d(len(i))
                                                          for i in inds])
        self.track_stats = False

    def forward(self, x):
        out_list = []
        if self.track_stats and\
           is_tracked(self.track_for, self._num_layer, self._num_block, 1):
            out = F.relu(self.bn1(self.conv1(x)))
            key = (self._num_layer, self._num_block, 1)
            # next_key = (self._num_layer, self._num_block, 2)
            # if next_key in self.indices.keys:
            #     next_inds = self.indices[next_key]
            #     tr = out.clone().detach()
            #     ret = inds_at_t(tr.cpu().numpy())
            #     inds = np.int32(ret["inds"])
            #     w = self.conv2.weight.clone().detach()
            #     batch_inds_for_labels = [np.where(self.labels == i)[0] for i in range(10)]
            #     topk = 10
            #     checks = []
            #     for label, b_inds in enumerate(batch_inds_for_labels):
            #         checks.append([])
            #         batch_i = tr[b_inds]
            #         inds_i = inds[b_inds, :topk]
            #         grid = make_grid(inds_i)
            #         shape = [*inds_i.shape, *out.shape[2:]]
            #         # batch for label i along indices with highst l1 norm
            #         bb = batch_i[grid[:, 0], grid[:, 1]].reshape(*shape)
            #         # inds differ for each element of batch
            #         for bb_, ii_ in zip(bb, inds_i):
            #             res_sum = []
            #             # NOTE:
            #             # for each instance we check the value of outputs on
            #             # each row (corresponding to each label) of next_inds
            #             # res_sum[i == true_label] > res_sum [i != true_label]
            #             for i, next_i in enumerate(next_inds):
            #                 bn_params = {"running_mean": self.bn2.running_mean[next_i],
            #                              "running_var": self.bn2.running_var[next_i],
            #                              "weight": self.bn2.weight[next_i],
            #                              "bias": self.bn2.bias[next_i]}
            #                 res = F.conv2d(bb_.unsqueeze(0), w[:, ii_][next_i],
            #                                stride=(1, 1), padding=(1, 1))
            #                 res = F.relu(F.batch_norm(res, **bn_params))
            #                 res_sum.append(res.view(len(next_i), -1).sum(1))
            #             checks[-1].append(torch.stack(res_sum).sum(1).argmax() == label)
            ret = inds_at_t(out.detach().cpu().numpy())
            inds, sum, csum = np.int32(ret["inds"]), ret["sum"], ret["cumsum"]
            # NOTE: We should collect by NEXT index
            for i, label in enumerate(self.labels):
                self.order_stats[label][key].extend([("inds", inds[i]),
                                                     ("sum", sum[i]),
                                                     ("csum", csum[i])])
        elif hasattr(self, "_num_layer") and hasattr(self, "indices") and\
             (self._num_layer, self._num_block, 1) in self.indices.keys:
            inds = self.indices[self._num_layer, self._num_block, 1]
            for c, b, i in zip(self.convlist1, self.bnlist1, inds):
                out_list.append(F.relu(b(c(x))))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        if self.track_stats and\
           is_tracked(self.track_for, self._num_layer, self._num_block, 2):
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            key = (self._num_layer, self._num_block, 2)
            ret = inds_at_t(out.detach().cpu().numpy())
            inds, sum, csum = np.int32(ret["inds"]), ret["sum"], ret["cumsum"]
            for i, label in enumerate(self.labels):
                self.order_stats[label][key].extend([("inds", inds[i]),
                                                     ("sum", sum[i]),
                                                     ("csum", csum[i])])
        elif hasattr(self, "_num_layer") and hasattr(self, "indices") and\
             (self._num_layer, self._num_block, 2) in self.indices.keys:
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
        self.track_stats = False
        self.track_for = "all"

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
        if self.track_stats:
            key = (0, 0, 0)
            if key in self.track_for:
                order = get_order(out)
                for k, v in order.items():
                    for i, label in enumerate(self.labels):
                        self.order_stats[label][key].append([k, v[i]])
        out = self.layer1(out)                  # (128, 16, 32, 32)
        out = self.layer2(out)                  # (128, 32, 16, 16)
        out = self.layer3(out)                  # (128, 64, 8, 8)
        return out

    def tail(self, x):                          # (128, 32, 16, 16)
        out = F.avg_pool2d(x, x.size()[3])      # (128, 64, 1, 1)
        out = out.view(out.size(0), -1)         # (128, 64)
        if self._dropout:
            return self.linear(F.dropout(out, self._dropout))
        else:
            return self.linear(out)                  # (128, 10)

    def forward(self, x):
        return self.tail(self.head(x))

    # def forward(self, x, path, k=1, target_var=None):
    #     if path == path_dict['CSG']:
    #         return self.forward_csg(x)
    #     elif path == path_dict['VAL']:
    #         return self.forward_csg_val_alt(x, target_var, k)
    #     else:
    #         return self.forward_std(x)


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
