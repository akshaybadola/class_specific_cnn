import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


def is_tracked(track_for, a, b, c):
    return any([t[0] == a and t[1] == b and t[2] == c for t in track_for])


def get_order(ar):
    temp = ar.detach().cpu().numpy()
    x, y = temp.shape[0:2]
    temp_sum = temp.reshape(x, y, -1).sum(2)
    order = np.argsort(temp_sum)[:, ::-1]
    return {"sum": temp_sum, "inds": order}


def make_grid(inds):
    """Make a grid for efficient indexing of n-d tensor from 2d inds.

    Example:
        inds = np.array([[0, 8, 4, 3], [2, 3, 4, 1], [3, 4, 1, 1]])
        grid = make_grid(inds)
        t = torch.randn(64, 64, 8, 8)
        shape = [*inds.shape, *t.shape[2:]]
        t[grid[:, 0], [:, 1]].reshape(*shape)[0] == tr[0, inds[0]]
        t[grid[:, 0], [:, 1]].reshape(*shape)[1] == tr[1, inds[1]]
        t[grid[:, 0], [:, 1]].reshape(*shape)[2] == tr[2, inds[2]]

    """
    x, y = inds.shape
    return np.array([*zip(np.repeat(np.arange(x), y).reshape(-1),
                          inds.reshape(-1))])


def inds_at_t(ar, t=0.9):
    """Get indices for threshold `t` for a given tensor

    The tensor is summed along last axes and the first axis is assumed to be the
    batch size. The sum (and the indices) is computed along the second axis.

    Args:
        t: threshold of sum of contributions of indices

    """
    x, y = ar.shape[0:2]
    temp_sum = ar.reshape(x, y, -1).sum(2)
    normalized = temp_sum / temp_sum.sum(1).reshape(x, 1)
    sorted_inds = np.argsort(temp_sum)[:, ::-1]
    grid = np.array([*zip(np.repeat(np.arange(x), y).reshape(-1), sorted_inds.reshape(-1))])
    sorted_norm = normalized[grid[:, 0], grid[:, 1]].reshape(x, y)
    sorted_cumsum = sorted_norm.cumsum(1)
    return {"inds": sorted_inds,
            "sum": temp_sum,
            "grid": grid,
            "cumsum": sorted_cumsum,
            "k": int((sorted_cumsum < t).sum(1).mean())}


def conv_with_w_at_t(tr, w, t=0.9):
    """Get result of convolution of tr with weights :code:`w` with threshold :code:`t`.

    :code:`w` are convolution weights and usually resricted along a set of input
    indices.  The result is calculated only for indices where total contribution
    of input w.r.t. l_1 norm is > t.

    Args:
        tr: Input tensor
        w: weights for multiplication
        t: threshold of sum of contributions of indices

    """
    ret = inds_at_t(tr.detach().cpu().numpy(), t)
    inds = ret["inds"]
    sum = ret["sum"]
    grid = ret["grid"]
    k = ret["k"]
    # This will differ according to class c
    least = grid.reshape(tr.shape[0], -1, 2)[:, k:, :].reshape(-1, 2)
    tr[least[:, 0], least[:, 1]] = 0
    tt = F.conv2d(tr, w, stride=(1, 1), padding=(1, 1))
    return {"k": k, "inds": inds, "sum": sum, "tt": tt}


def conv_with_w_for_inds(tr, w, inds):
    """Get result of convolution of tr with weights :code:`w` restricted to :code:`inds`.

    The input and weight tensors are restricted along indices given by :code:`inds`

    Args:
        w: convolution weights
        tr: Input tensor
        inds: Indices along which to restrict the operation
    """
    # This will differ according to class c
    x, y = tr.shape[0:2]
    # FIXME: Not sure this is correct
    grid = np.array([*zip(np.repeat(np.arange(x), y).reshape(-1), inds.reshape(-1))])
    mask = torch.zeros_like(tr).bool()
    mask[grid[:, 0], grid[:, 1]] = 1
    return F.conv2d(tr*mask, w, stride=(1, 1), padding=(1, 1))


# def save_plot(output):
#     ix = 1
#     total = 0
#     each = 4
#     if not os.path.exists('fmaps'):
#         os.makedirs('fmaps')
#     while total < output.shape[0]:
#         for _ in range(each):
#             for _ in range(each):
#                 if total < output.shape[0]:
#                     ax = plt.subplot(each, each, ix)
#                     ax.set_xticks([])
#                     ax.set_yticks([])
#                     plt.imshow(output[total, :, :].cpu(), cmap='gray')
#                 ix += 1
#                 total += 1
#         plt.savefig("fmaps/"+str(total//each**2)+".jpg")
#         print(ix, total)
#         ix = 1
