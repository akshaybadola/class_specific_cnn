import argparse
import pickle
import json
import copy

import pandas as pd
import numpy as np

import info
from class_selectivity_index import selectivity
from functions import collect_outputs_at_submodules, get_model, get_model_and_data
import finetune



def display_psi():
    with open("all_metrics.pkl", "rb") as f:
        all_metrics = pickle.load(f)

    layers = dict([(x, i+1) for i, x in enumerate(all_metrics['regular'].keys())])

    metrics = {}
    for k in all_metrics:
        metrics[k] = {}
        for layer in all_metrics[k]:
            metrics[k][layers[layer]] = np.mean([x[1] for x in all_metrics[k][layer].values()])

    # Filter shapes in resnet20
    num_filters = np.concatenate([np.repeat(16, 4), np.repeat(32, 3), np.repeat(64, 3)])

    df = pd.DataFrame.from_dict(metrics)
    for k in df:
        df[k] = (num_filters - df[k])/num_filters
    df.loc["mean"] = df.mean()
    print(df.to_markdown(tablefmt="grid"))


def calc_score_probs(probs, num_classes):
    score = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                a = probs[i]
                b = probs[j]
                ab = (probs[i] + probs[j])
                ab = ab/ab.sum()
                score.append(info.mutual_info(a, b, ab))
    return np.mean(score)


def print_info():
    with open("all_mi.pkl", "rb") as f:
        all_mi = pickle.load(f)
    score = {}
    for w in all_mi:
        temp = calc_score_probs(all_mi, w)
        layers = dict([(x, i+1) for i, x in enumerate(temp.keys())])
        score[w] = dict(zip(layers.values(), temp.values()))
    df = pd.DataFrame.from_dict(score)
    df.loc["mean"] = df.mean()
    return df


def get_mu_for_layer(model, data, dataloaders, k, num_classes, submodule):
    result, output_matrix, corr_matrix = collect_outputs_at_submodules(
        model, dataloaders, num_classes, submodule)
    mu = {}
    for c in output_matrix:
        sel = selectivity.calculate_selectivity_numpy({"vals": output_matrix[c]})
        sel = [*sel.values()][0]
        sel = np.array([x[1] for x in sel if x[0]])
        mu[c] = np.mean(sel)
    return mu


def calc_mu(output_matrix):
    mu = {}
    for c in output_matrix:
        sel = selectivity.calculate_selectivity_numpy({"vals": output_matrix[c]})
        sel = [*sel.values()][0]
        sel = np.array([x[1] for x in sel if x[0]])
        mu[c] = np.mean(sel)
    return mu


def calc_psi_at_fraction(output_matrix, k):
    inds = {}
    psi = {}
    sorted_output = {}
    for i in output_matrix:
        shape = output_matrix[i].shape
        inds[i] = output_matrix[i].argsort(1)
        inds[i] = np.flip(inds[i], 1)
        sorted_output[i] = output_matrix[i][np.repeat(np.arange(shape[0]), shape[1]),
                                            inds[i].flatten()].reshape(shape)
        # To get the k_th fraction weight
        sorted_output[i] = (sorted_output[i].T / sorted_output[i].sum(1)).T
        psi[i] = (np.median((sorted_output[i].cumsum(1) < k).sum(1)),
                      np.mean((sorted_output[i].cumsum(1) < k).sum(1)))
    return psi


def calc_pairwise_mi(output_matrix, k):
    minlength = output_matrix[0].shape[1]
    mi = {}
    inds = {}
    counts = {}
    top_counts = {}
    probs = []
    selected = []
    for i in output_matrix:
        inds[i] = output_matrix[i].argsort(1)
        inds[i] = np.flip(inds[i], 1)
        counts[i] = np.bincount(inds[i][:, :k].flatten(), minlength=minlength)
        top_counts[i] = counts[i].argsort()[::-1][:k]
        probs.append(counts[i]/counts[i].sum())
        selected.append(np.int32(probs[i] != 0))
    score_prob = []
    score_select = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                score_prob.append(info.mutual_info(probs[i], probs[j], probs[i] + probs[j]))
                score_select.append(info.mutual_info(selected[i], selected[j], selected[i] | selected[j]))
    mi["prob"] = np.mean(score_prob)
    mi["select"] = np.mean(score_select)
    return mi


def get_indices_final_layer(model, data, dataloaders, k, num_classes, submodule):
    result, output_matrix, corr_matrix = collect_outputs_at_submodules(
        model, dataloaders, num_classes, submodule)
    inds = {i: None for i in range(num_classes)}
    counts = {i: None for i in range(num_classes)}
    top_inds = {i: None for i in range(num_classes)}
    minlength = output_matrix[0].shape[1]
    for i in range(num_classes):
        inds[i] = output_matrix[i].argsort(1)
        inds[i] = np.flip(inds[i], 1)
        counts[i] = np.bincount(inds[i][:, :k].flatten(), minlength=minlength)
        top_inds[i] = counts[i].argsort()[::-1][:k]
    return top_inds


def maybe_k(k):
    try:
        k = int(k)
    except ValueError:
        k = float(k)            # type: ignore
    return k


def get_model_and_data_from_args(args):
    if args.model not in {"resnet20", "resnet50"}:
        raise NotImplementedError("Not implemented for other than Resnet20,50. "
                                  "You'll have to add the module names for other models to extract"
                                  " the indices yourself")
    if args.model == "resnet20" and not args.weights:
        raise ValueError("Model weights must be given for resnet20 as it's not there in torchvision")
    model, data, dataloaders, num_classes = get_model_and_data(args.model,
                                                               args.weights,
                                                               args.batch_size)
    return model, data, dataloaders, num_classes


def do_finetune(args):
    model, data, dataloaders, num_classes = get_model_and_data_from_args(args)
    k = maybe_k(args.k)
    if args.finetune_method == "decomposed":
        if not args.inds_file:
            if args.model == "resnet20":
                indices = get_indices_final_layer(model, data, dataloaders, k, num_classes, "layer3.2")
            elif args.model == "resnet50":
                indices = get_indices_final_layer(model, data, dataloaders, k, num_classes, "avgpool")
        else:
            with open(args.inds_file) as f:
                indices = json.load(f)
            keys = [*indices.keys()]
            keys.sort()         # just in case
            indices = np.array([indices[k] for k in keys])
        # Load model again with different forward function
        model = get_model(args.model, args.weights, num_classes, indices=indices)
        model.forward = model.forward_decomposed
        # Need to initialize final weights. Earlier ones are random
        model._init_final_weights()
        print("Validating the model after loading decomposed indices")
        finetune.validate(model, dataloaders["val"], args.gpu, print_only=True)
        finetune.finetune(model, args.model, dataloaders, args.num_epochs, args.lr, gpu=args.gpu)
    elif args.finetune_method == "selectivity":
        pass
    else:
        raise ValueError("Unknown finetune method")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", help="Command to run")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--weights", help="Model weights file")
    parser.add_argument("--batch-size", "-b", type=int, default=128,
                        help="Trainig batch size for finetuning")
    parser.add_argument("--num-epochs", "-n", type=int, help="Number of epochs for finetuning",
                        default=100)
    parser.add_argument("--lr", type=float, help="Learning rate",
                        default=2e-04)
    parser.add_argument("--finetune-method", choices=["decomposed", "selectivity"])
    parser.add_argument("--layer-name", default="layer3.2",
                        help="Submodule (layer) name to calculate metrics. "
                        "E.g. for resnet20 final conv layer is \"layer3.2\"")
    parser.add_argument("--inds-file")
    parser.add_argument("-k", help="Threshold for choosing top inds. Can be float or int. "
                        "Used only when --inds-file is not given.")
    parser.add_argument("--gpu", "-g", type=int, default=0, help="GPU to use for finetuning")
    args = parser.parse_args()
    if args.cmd == "calc_mi":
        model, data, dataloaders, num_classes = get_model_and_data_from_args(args)
        k = maybe_k(args.k)
        result, output_matrix, corr_matrix = collect_outputs_at_submodules(
            model, dataloaders, num_classes, args.layer_name)
        print(calc_pairwise_mi(output_matrix, k))
    if args.cmd == "calc_psi":
        model, data, dataloaders, num_classes = get_model_and_data_from_args(args)
        k = maybe_k(args.k)
        result, output_matrix, corr_matrix = collect_outputs_at_submodules(
            model, dataloaders, num_classes, args.layer_name)
        print(calc_psi_at_fraction(output_matrix, k))
    elif args.cmd == "calc_mu":
        model, data, dataloaders, num_classes = get_model_and_data_from_args(args)
        result, output_matrix, corr_matrix = collect_outputs_at_submodules(
            model, dataloaders, num_classes, args.layer_name)
        print(calc_mu(output_matrix))
    elif args.cmd == "finetune":
        do_finetune(args)
