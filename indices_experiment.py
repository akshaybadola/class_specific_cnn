import argparse
import importlib
import numpy as np
import torch
import torch.nn.functional as F

import info

from functions import get_model_data, get_results_at_key



def main(key, k):
    model, data, dataloaders, num_classes = get_model_data()
    result, output_matrix, corr_matrix = get_results_at_key(model, dataloaders, num_classes, key)

    inds = []
    minlength = corr_matrix[0].shape[0]
    counts = []
    for i in range(num_classes):
        counts.append(np.bincount(corr_matrix[i].argsort(1)[:, -k:].flatten(),
                                  minlength=minlength))
    probs = []
    for i in range(num_classes):
        probs.append(np.int32(counts[i] > np.mean(counts[i])))

    # entropy = []
    # score = []
    # for i in range(num_classes):
    #     for j in range(num_classes):
    #         a = np.zeros_like(range(minlength))
    #         b = np.zeros_like(range(minlength))
    #         a[inds[i]] = 1
    #         b[inds[j]] = 1
    #         entropy.append(sp.stats.entropy(a, b))
    #         score.append(mutual_info_score(a, b))
    # print(f"Mutual Info Score at layer {key}", np.mean(score))
    score = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                score.append(info.mutual_info(probs[i], probs[j], probs[i] | probs[j]))
    print(f"Mutual Info Score at layer {key}", np.mean(score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("key")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    key = map(int, args.key.split(","))
    main(tuple(key), args.k)
