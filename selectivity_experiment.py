import argparse

import torch

from common_pyutil.monitor import Timer

import resnet_a_with_relu
import resnet_b
import finetune
import util
import dataset


def get_model_and_data(model_name, batch_size):
    if model_name == "resnet20":
        model = resnet_a_with_relu.resnet20(10)
        weights = torch.load("resnet20-12fca82f.th", map_location="cpu")
        util.load_fixing_names(model, weights)
        data, dataloaders, num_classes = dataset.get_data("cifar-10",
                                                          {"train": batch_size,
                                                           "val": 64,
                                                           "test": 64},
                                                          {"train": 12,
                                                           "val": 12,
                                                           "test": 12})
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


def main(model_name, batch_size, alpha, lr):
    model, data, dataloaders, num_classes = get_model_and_data(model_name, batch_size)
    if model_name == "resnet50":
        print_frequency = 1
    else:
        print_frequency = 10
    finetune.finetune_with_selectivity_regularizer(model, model_name,
                                                   dataloaders, num_classes, alpha,
                                                   lr=lr,
                                                   print_frequency=print_frequency)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Name of the model")
    parser.add_argument("--batch-size", "-b", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-04)
    args = parser.parse_args()
    main(args.model, args.batch_size, args.alpha, args.lr)
