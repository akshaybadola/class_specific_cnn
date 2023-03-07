import importlib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

import util
import dataset
import hooks
import original_resnet_a


def get_model_data():
    model = original_resnet_a.resnet20(10)
    weights = torch.load("resnet20-12fca82f.th", map_location="cpu")
    util.load_fixing_names(model, weights)
    if util.have_cuda():
        model = model.cuda()
    model = model.eval()
    data, dataloaders, num_classes = dataset.get_data("cifar-10",
                                                      {"train": 128,
                                                       "val": 64,
                                                       "test": 64},
                                                      {"train": 12,
                                                       "val": 12,
                                                       "test": 12})
    return model, data, dataloaders, num_classes


def get_results_at_key(model, dataloaders, num_classes, key):
    results, labels = hooks.get_outputs_at(model, "resnet20", dataloaders, key)
    key = [*results.keys()][0]
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
