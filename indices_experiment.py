from typing import Tuple, List
import json
import pickle
import argparse
import importlib
import numpy as np
import torch
import torch.nn.functional as F

import hooks
import info

from functions import get_model_data, get_results_at_key



def main(keys, k, save=True):
    model, data, dataloaders, num_classes = get_model_data()
    if keys == "all":
        keys = hooks.get_keys(model, ["layer1", "layer2", "layer3"], layer_only=True)
    mutual_info = {}
    inds = {}
    eigvals = {}
    for key in keys:
        result, output_matrix, corr_matrix = get_results_at_key(model, dataloaders, num_classes, key)

        minlength = corr_matrix[0].shape[0]
        counts = []
        for i in range(num_classes):
            counts.append(np.bincount(corr_matrix[i].argsort(1)[:, -k:].flatten(),
                                      minlength=minlength))
        probs = []
        for i in range(num_classes):
            mean_counts = np.mean(counts[i][counts[i] > 0])
            probs.append(np.int32(counts[i] >= mean_counts))

        inds[key] = []
        for i in range(num_classes):
            inds[key].append(np.where(probs[i] == 1)[0].tolist())
        score = []
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:
                    score.append(info.mutual_info(probs[i], probs[j], probs[i] | probs[j]))
        print(f"Mutual Info Score at layer {key}", np.mean(score))
        mutual_info[key] = float(np.mean(score))

        x = np.matmul(result.T, result)
        v = np.linalg.eigvals(x)
        eigvals[key] = {}
        eigvals[key]["whole"] = v
        for i in range(num_classes):
            eigvals[key][i] = np.linalg.eigvals(corr_matrix[i])

        if save:
            with open(f"eigvals_{key}.pkl", "wb") as f:
                pickle.dump(eigvals, f)
            # with open(f"probs_{key}.pkl", "wb") as f:
            #     pickle.dump(probs, f)
            # with open(f"counts_{key}.pkl", "wb") as f:
            #     pickle.dump(counts, f)
            # with open(f"mutual_info_{key}.pkl", "wb") as f:
            #     pickle.dump(mutual_info, f)
            # with open(f"inds_{key}.pkl", "wb") as f:
            #     pickle.dump(inds, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("keys")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("-s", action="store_true")
    args = parser.parse_args()
    if args.keys == "all":
        keys = args.keys
    else:
        keys: List[Tuple[int, int]] = [tuple(map(int, key.split(",")))
                                       for key in args.keys.split(";")]
    main(keys, args.k, args.s)
