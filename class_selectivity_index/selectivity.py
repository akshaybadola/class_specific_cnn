from typing import List, Dict, Tuple

import torch
import numpy as np


def prepare_activations(activations, hidden_outputs, labels, num_classes):
    for name in activations:
        ho_shape = hidden_outputs[name].shape
        temp = hidden_outputs[name].view(*ho_shape[:2], -1).sum(-1)
        for i in range(num_classes):
            class_activation = temp[labels == i]
            filterwise_sum = class_activation.sum(0)
            if activations[name][i] is None:
                activations[name][i] = filterwise_sum
            else:
                activations[name][i] += filterwise_sum


def calculate_selectivity_numpy(activations):
    result: Dict[str, List[Tuple[int, float]]] = {}
    for name in activations:
        print(f'Calculating class selectivity index for {name}')
        layer_act = activations[name]

        if isinstance(layer_act, list):
            layer_act = np.array(layer_act)

        num_classes, num_units = layer_act.shape

        result[name] = []
        for unit in range(num_units):
            mm_ind = layer_act.T[unit].argmax()
            mu_max = layer_act.T[unit, mm_ind]
            mu_min = np.delete(layer_act.T[unit], mm_ind).mean()
            result[name].append((mm_ind, (mu_max - mu_min)/(mu_max + mu_min + 1e-07)))
    return result


def calculate_selectivity_torch(activations):
    result: Dict[str, List[Tuple[int, float]]] = {}
    for name in activations:
        print(f'Calculating class selectivity index for {name}')
        layer_act = activations[name]

        if isinstance(layer_act, list):
            layer_act = torch.stack(layer_act)

        num_classes, num_units = layer_act.shape

        result[name] = []
        for unit in range(num_units):
            t = layer_act.T
            mm_ind = t[unit].argmax()
            inds = torch.arange(num_classes)
            rest_inds = torch.cat([inds[:mm_ind], inds[mm_ind+1:]])
            mu_max = t[unit, mm_ind]
            mu_min = t[unit, rest_inds].mean()
            result[name].append((mm_ind, (mu_max - mu_min)/(mu_max + mu_min + 1e-07)))
    return result


def calculate_selectivity_torch_new(activations, num_labels):
    result: Dict[str, List[Tuple[int, float]]] = {}
    non_zero = num_labels != 0
    for name in activations:
        print(f'Calculating class selectivity index for {name}')
        layer_act = activations[name]

        if isinstance(layer_act, list):
            layer_act = torch.stack(layer_act)
        if isinstance(layer_act, np.ndarray):
            layer_act = torch.tensor(layer_act)

        num_classes, num_units = layer_act.shape

        result[name] = []
        t = layer_act[non_zero].T / num_labels[non_zero]
        result[name] = 0
        sorted = t.sort(1, descending=True)[0]
        mu_max = sorted[:, 0]
        mu_min = sorted[:, 1:].mean(1)
        result[name] = (mu_max - mu_min)/(mu_max + mu_min + 1e-07)
        result[name] = result[name].mean()
    return result


def selectivity_regularizer(activations, num_labels):
    result: Dict[str, float] = {}
    non_zero = num_labels != 0
    for name in activations:
        layer_act = activations[name]

        if isinstance(layer_act, list):
            layer_act = torch.stack(layer_act)

        num_classes, num_units = layer_act.shape
        t = layer_act[non_zero].T / num_labels[non_zero]
        result[name] = 0
        sorted = t.sort(1, descending=True)[0]
        mu_max = sorted[:, 0]
        mu_min = sorted[:, 1:].mean(1)
        result[name] = (mu_max - mu_min)/(mu_max + mu_min + 1e-07)
        result[name] = result[name].mean()
        # inds = torch.arange(t.shape[1])
        # for unit in range(num_units):
        #     mm_ind = t[unit].argmax()
        #     rest_inds = torch.cat([inds[:mm_ind], inds[mm_ind+1:]])
        #     mu_max = t[unit, mm_ind]
        #     mu_min = t[unit, rest_inds].mean()
        #     result[name] += (mu_max - mu_min)/(mu_max + mu_min + 1e-07)
        # result[name] /= num_units
    return result
