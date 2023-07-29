#!/usr/bin/env python

# Adapted from https://github.com/1Konny/class_selectivity_index/

from typing import Dict, List, Optional
import os
import sys
import argparse

import numpy as np

from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets


have_cuda = torch.cuda.is_available()


def get_model_and_data(img_root, batch_size=25, imgs_per_class=0, model_weights=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    num_classes = 1000
    iters_per_class = imgs_per_class//batch_size
    assert imgs_per_class/batch_size == float(iters_per_class)

    num_workers = 6
    dset = datasets.ImageFolder(img_root, transform=transform)
    loader = DataLoader(dset, batch_size=batch_size,
                        drop_last=False, shuffle=False, num_workers=num_workers)
    print('# of images :', len(dset))
    model = models.resnet50(pretrained=True)
    model.eval()
    model.cuda()
    return model, loader, num_classes, iters_per_class


# and set hooks to extract hidden activations across layers
def register_hook(model, activations, hidden_outputs):
    def named_hook(name):
        def hook(module, input, output):
            hidden_outputs[name] = output
        return hook

    for name in activations.keys():
        if name in model._modules:
            if 'layer' in name:
                relu = getattr(model._modules[name][-1], "relu1", None) or\
                    getattr(model._modules[name][-1], "relu")
                if relu:
                    relu.register_forward_hook(named_hook(name))
                else:
                    raise AttributeError(f"relu not found in layer {name}")
            elif 'pool' in name:
                model._modules[name].register_forward_hook(named_hook(name))
            print(f'Registered for {name}')


def calculate_selectivity(activations):
    result = {}
    with torch.no_grad():
        for name in activations:
            print(f'Calculating class selectivity index for {name}')
            layer_act = activations[name]

            if isinstance(layer_act, list):
                layer_act = torch.tensor(np.array(layer_act))
            if isinstance(layer_act, np.ndarray):
                layer_act = torch.tensor(layer_act)
            num_classes, num_neuron = layer_act.size()
            dead_neuron_class = torch.tensor(num_classes)
            dead_neuron_confidence = torch.tensor(0.)

            selected_class = []
            selectivity_index = []
            for neuron_idx in range(num_neuron):
                neuron_act = layer_act[:, neuron_idx]
                # In the case of mean activations of a neuron are all zero across whole classes
                # Simply consider that neuron as dead neuron.
                if neuron_act.nonzero().size(0) == 0:
                    class_selected = dead_neuron_class
                    class_confidence = dead_neuron_confidence
                else:
                    class_selected = neuron_act.argmax()
                    mu_max = neuron_act[class_selected]
                    mu_mmax = (neuron_act.sum()-mu_max).div(num_classes-1)
                    class_confidence = (mu_max-mu_mmax).div(mu_max+mu_mmax)
                selected_class.append(class_selected)
                selectivity_index.append(class_confidence)
                if not (neuron_idx+1) % (num_neuron // 100):
                    print(f"{np.ceil(neuron_idx/num_neuron*100)} percent done")
            selected_class = torch.stack(selected_class, 0)
            selectivity_index = torch.stack(selectivity_index, 0)
            result[name] = {'selected_class': selected_class, 'selectivity_index': selectivity_index}
    return result


def plot_selectivity_result(result, prefix: str, output_dir: str):
    """Plot class selectivity distributions as a function of depth"""
    num_plots = len(result)
    colormap = plt.cm.jet
    plt.figure(figsize=(10, 10))
    axes = plt.gca()
    axes.set_prop_cycle(color=[colormap(i) for i in np.linspace(0, 0.9, num_plots)])

    labels = []
    for name in result:
        selectivity_index = result[name]['selectivity_index']
        selectivity_index_hist = np.histogram(selectivity_index*100, bins=100, normed=True)
        x = np.arange(len(selectivity_index_hist[0]))/len(selectivity_index_hist[0])
        y = selectivity_index_hist[0]
        plt.fill_between(x, y, step="pre", alpha=0.6)
        plt.plot(x, y)
        labels.append(name)

    plt.legend(labels, ncol=1, loc='upper right',
               columnspacing=2.0, labelspacing=1,
               handletextpad=0.5, handlelength=1.5,
               fancybox=True, shadow=True)
    plt.ylabel('PDF', fontsize=15, labelpad=15)
    plt.xlabel('Selectivity Index', fontsize=15, labelpad=15)

    figname = os.path.join(output_dir, f'{prefix}_histogram.png')
    plt.savefig(figname)
    plt.show()


def get_class_conditional_sum_activation(model, pool_layer, loader, num_classes):
    """Get Class Conditional Mean Activation per layer

    Args:
        model: Model
        pool_layer: One of avgpool or split_pool
        loader: dataloader
        num_classes: number of classes

    Get sum of activation of each filter per class at the end of each layer.
    We convert fp32 torch tensor to fp64 numpy array and sum it up, and
    also return number of instances per class.

    """
    if pool_layer == "split_pool":
        activations = dict(layer1=[], layer2=[], layer3=[], layer4=[],
                           split_pool=[])
    elif pool_layer == "avgpool":
        activations = dict(layer1=[], layer2=[], layer3=[], layer4=[],
                           avgpool=[])
    else:
        raise ValueError(f"Unknown pool layer {pool_layer}")

    hidden_outputs: Dict[str, torch.Tensor] = {}
    register_hook(model, activations, hidden_outputs)

    model = model.eval()

    if not hasattr(model, "layer4"):
        activations.pop("layer4")
    if not hasattr(model, "avgpool") and "avgpool" in activations:
        activations.pop("avgpool")
    if have_cuda:
        model = model.cuda()

    it = iter(loader)
    class_conditional_activations: Dict[str, List[Optional[np.ndarray]]] = {}
    for name in activations:
        class_conditional_activations[name] = [None for _ in range(num_classes)]
    total_labels = np.zeros(num_classes, dtype="int64")
    with torch.no_grad():
        for batch_num, batch in enumerate(loader):
            imgs, labels = it.__next__()
            if have_cuda:
                imgs = imgs.cuda()
            model(imgs)
            total_labels += torch.bincount(labels, minlength=num_classes).detach().cpu().numpy()
            for name in activations:
                ho_shape = hidden_outputs[name].shape
                temp = hidden_outputs[name].view(*ho_shape[:2], -1).sum(-1)
                for i in range(num_classes):
                    class_activation = temp[labels == i]
                    filterwise_sum = class_activation.sum(0).detach().cpu().numpy()
                    if class_conditional_activations[name][i] is None:
                        class_conditional_activations[name][i] = filterwise_sum.astype("float64")
                    else:
                        class_conditional_activations[name][i] += filterwise_sum.astype("float64")
            if not (batch_num+1) % 100:
                print(f"{batch_num+1} batches done")
        return class_conditional_activations, total_labels


def load_activations(layer_names, output_dir):
    activations = {}
    for name in layer_names:
        activations_path = os.path.join(output_dir, name+'_activations.pth')
        activations[name] = torch.load(activations_path)
    return activations


def save_activations(activations, output_dir):
    for name in activations:
        activations_path = os.path.join(output_dir, name+'_activations.pth')
        torch.save(activations[name], activations_path)
        print(f'Saved {activations_path}')


def save_selectivity(result, output_dir):
    for name in activations:
        result_path = os.path.join(output_dir, name+'_selectivity_results.pth')
        torch.save(result[name], result_path)
        print(f'Saved {result_path}')


def load_selectivity(layer_names, output_dir):
    selectivity = {}
    for name in layer_names:
        selectivity_path = os.path.join(output_dir, name+'_selectivity_results.pth')
        selectivity[name] = torch.load(selectivity_path)
    return selectivity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    activations = get_activations(args.img_root, args.output_dir, args.save_outputs)
    selectivity = calculate_selectivity(activations, args.output_dir, args.save_outputs)
    if args.plot:
        plot_selectivity_result(selectivity, args.output_dir)
