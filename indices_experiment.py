from typing import Tuple, List, Dict
import os
import argparse
import json
import numpy as np

import hooks
import info

from functions import get_model_and_data, get_outputs_at_submodule


def calculcate_metrics(model, dataloaders, num_classes, key, k):
    result, output_matrix, corr_matrix = get_outputs_at_submodule(model, dataloaders, num_classes, key)

    eigvals = {}
    minlength = corr_matrix[0].shape[0]

    counts = []
    for i in range(num_classes):
        counts.append(np.bincount(corr_matrix[i].argsort(1)[:, -k:].flatten(),
                                  minlength=minlength))
    probs = []
    for i in range(num_classes):
        mean_counts = np.mean(counts[i][counts[i] > 0])
        probs.append(np.int32(counts[i] >= mean_counts))

    inds = []
    for i in range(num_classes):
        inds.append(np.where(probs[i] == 1)[0].tolist())

    score = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                score.append(info.mutual_info(probs[i], probs[j], probs[i] | probs[j]))
    print(f"Mutual Info Score at layer {key}", np.mean(score))
    mutual_info = float(np.mean(score))

    x = np.matmul(result.T, result)
    v = np.linalg.eigvals(x)
    eigvals = {}
    eigvals["whole"] = v
    for i in range(num_classes):
        eigvals[i] = np.linalg.eigvals(corr_matrix[i])
    return {"counts": counts, "probs": probs, "inds": inds, "score": score,
            "eigvals": eigvals, "mutual_info": mutual_info}


def calculate_and_print_metrics(model, dataloaders, num_classes, key):
    result, output_matrix, corr_matrix = get_outputs_at_submodule(model, dataloaders, num_classes, key)
    metrics = {}
    o_inds = {}
    sorted_output = {}
    for i in output_matrix:
        o_shape = output_matrix[i].shape
        o_inds[i] = output_matrix[i].argsort(1)
        o_inds[i] = np.flip(o_inds[i], 1)
        sorted_output[i] = output_matrix[i][np.repeat(np.arange(o_shape[0]), o_shape[1]),
                                            o_inds[i].flatten()].reshape(o_shape)
        # To get the 50% weight
        sorted_output[i] = (sorted_output[i].T/ sorted_output[i].sum(1)).T
        metrics[i] = (np.median((sorted_output[i].cumsum(1) < .5).sum(1)),
                      np.mean((sorted_output[i].cumsum(1) < .5).sum(1)))
    print(metrics)
    return metrics


def get_influential_inds(model, dataloaders, num_classes, key):
    result, output_matrix, corr_matrix = get_outputs_at_submodule(model, dataloaders, num_classes, key)
    metrics = {}
    o_inds = {}
    sorted_output = {}
    for i in output_matrix:
        o_shape = output_matrix[i].shape
        o_inds[i] = output_matrix[i].argsort(1)
        o_inds[i] = np.flip(o_inds[i], 1)
        sorted_output[i] = output_matrix[i][np.repeat(np.arange(o_shape[0]), o_shape[1]),
                                            o_inds[i].flatten()].reshape(o_shape)
        # To get the 50% weight
        sorted_output[i] = (sorted_output[i].T/ sorted_output[i].sum(1)).T
        metrics[i] = (np.median((sorted_output[i].cumsum(1) < .5).sum(1)),
                      np.mean((sorted_output[i].cumsum(1) < .5).sum(1)))
    print(metrics)
    return metrics


def main(weights, keys, k, save=True):
    model, data, dataloaders, num_classes = get_model_and_data("resnet20", weights, 256)
    if keys == "all":
        keys = hooks.get_keys(model, ["layer1", "layer2", "layer3"], layer_only=True)
    for key in keys:
        calculate_and_print_metrics(model, dataloaders, num_classes, key)
        # counts, probs, inds, score, eigvals, mi =\
        #     calculcate_metrics(model, dataloaders, num_classes, key, k).values()
        # if save:
        #     with open(f"eigvals_{key}.pkl", "wb") as f:
        #         pickle.dump({key: eigvals}, f)
        #     with open(f"probs_{key}.pkl", "wb") as f:
        #         pickle.dump({key: probs}, f)
        #     with open(f"counts_{key}.pkl", "wb") as f:
        #         pickle.dump({key: counts}, f)
        #     with open(f"mutual_info_{key}.pkl", "wb") as f:
        #         pickle.dump({key: mi}, f)
        #     with open(f"inds_{key}.pkl", "wb") as f:
        #         pickle.dump({key: inds}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd")
    parser.add_argument("keys")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--save", "-s", action="store_true")
    args = parser.parse_args()

    if args.cmd == "calc_metrics":
        metrics: Dict[str, List] = {"resnet20-12fca82f.th": [],
                                    "resnet20-selectivity.pth": [],
                                    "resnet20-decomposed.pth": []}
        if not all (os.path.exists(x) for x in metrics.keys()):
            raise FileNotFoundError("Not all weights exist")
        if args.keys == "all":
            keys = "0,0 1,0 1,1 1,2 2,0 2,1 2,2 3,0 3,1 3,2".split(" ")
        else:
            keys = args.keys.split(" ")
        keys = [tuple(map(int, key.split(","))) for key in keys]
        for key in keys:
            for weights in metrics:
                model, data, dataloaders, num_classes = get_model_and_data("resnet20", weights, 256)
                metrics[weights].append(calculate_and_print_metrics(model, dataloaders,
                                                                    num_classes, key))
        with open(f"metrics_{args.keys}.json", "w") as f:
            json.dump(metrics, f)
    elif args.cmd == "get_indices":
        pass
    elif args.cmd == "finetune_selectivity":
        pass
    elif args.cmd == "finetune_decomposed":
        pass


    # if args.keys == "all":
    #     keys = args.keys
    # else:
    #     keys: List[Tuple[int, int]] = [tuple(map(int, key.split(",")))
    #                                    for key in args.keys.split(";")]
    # main(args.weights, keys, args.k, args.s)




