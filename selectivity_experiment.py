import argparse

import torch

from common_pyutil.monitor import Timer

import resnet_a_with_relu
import finetune
import util
import dataset


def main(alpha):
    model = resnet_a_with_relu.resnet20(10)
    weights = torch.load("resnet20-12fca82f.th", map_location="cpu")
    util.load_fixing_names(model, weights)
    data, dataloaders, num_classes = dataset.get_data("cifar-10",
                                                      {"train": 128,
                                                       "val": 64,
                                                       "test": 64},
                                                      {"train": 12,
                                                       "val": 12,
                                                       "test": 12})
    finetune.finetune_with_selectivity_regularizer(model, "resnet20",
                                                   dataloaders, num_classes, alpha)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "-b", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    main(args.alpha)
