from typing import Dict, List
import importlib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

import util
import dataset
import hooks
import resnet_a_with_relu as resnet_a
import resnet_b


from util import have_cuda


def forward_get_labels(model, loader, gpu=0):
    if have_cuda and gpu is not None:
        model = model.cuda(gpu)
    model = model.eval()
    model = model.cuda(gpu)
    labels = []
    total_iters = len(loader)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            imgs, _labels = batch
            if have_cuda and gpu is not None:
                imgs, _labels = imgs.cuda(gpu), _labels.cuda(gpu)
            labels.append(_labels.cpu().numpy())
            _ = model.forward(imgs)
            if (i + 1) % 100 == 0:
                print(f"Done {i+1} iterations out of {total_iters}")
    return labels


def get_model(model_name, weights_file, num_classes, **kwargs):
    if model_name == "resnet20":
        model = resnet_a.resnet20(num_classes, **kwargs)
        weights = torch.load(weights_file, map_location="cpu")
        util.load_fixing_names(model, weights)
    elif model_name == "resnet50":
        model = resnet_b.resnet50(num_classes, pretrained=True, **kwargs)
    return model


def get_model_and_data(model_name, weights_file, batch_size):
    if model_name == "resnet20":
        data, dataloaders, num_classes = dataset.get_data("cifar-10",
                                                          {"train": batch_size,
                                                           "val": 64,
                                                           "test": 64},
                                                          {"train": 12,
                                                           "val": 12,
                                                           "test": 12})
        model = resnet_a.resnet20(num_classes)
        weights = torch.load(weights_file, map_location="cpu")
        util.load_fixing_names(model, weights)
    elif model_name == "resnet50":
        data, dataloaders, num_classes = dataset.get_data("imagenet",
                                                          {"train": batch_size,
                                                           "val": batch_size,
                                                           "test": batch_size},
                                                          {"train": 32,
                                                           "val": 16,
                                                           "test": 16})
        model = resnet_b.resnet50(num_classes, pretrained=True)
    return model, data, dataloaders, num_classes


def collect_outputs_at_submodules(model, dataloaders, num_classes, key: str):
    results: Dict[str, List] = {key: []}
    module = model.get_submodule(key)
    handle = module.register_forward_hook(lambda *x: results[key].append(x[-1]))
    labels = forward_get_labels(model, dataloaders["train"])
    handle.remove()
    for i, x in enumerate(results[key]):
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        results[key][i] = x

    result = torch.cat(results[key]).cpu().numpy()
    labels = np.concatenate(labels)

    output_matrix = {}
    for c in range(num_classes):
        output_matrix[c] = result[labels == c]

    corr_matrix = []
    for c in range(num_classes):
        corr_matrix.append(np.matmul(output_matrix[c].T, output_matrix[c]))

    return result, output_matrix, corr_matrix


def plot_model_weights_eigenvalues(model, key):
    conv, _ = model.get_child(key)
    with torch.no_grad():
        x = conv._parameters['weight'].clone().detach().cpu().numpy()
        x = x.reshape(x.shape[0], -1)
        v = np.linalg.eigvals(np.matmul(x, x.T))
    v.sort()
    plt.plot(v, 'r.')
    plt.show()
