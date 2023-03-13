from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from common_pyutil.monitor import Timer

import util
from class_selectivity_index.resnet import register_hook
from class_selectivity_index.selectivity import (selectivity_regularizer,
                                                 prepare_activations,
                                                 calculate_selectivity_torch)


def validate(model, val_loader, gpu=0):
    timer = Timer(True)
    model = model.eval()
    if util.have_cuda():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    correct = 0
    total = 0
    preds = {"preds": [], "labels": []}
    vfunc = getattr(model, "forward", None) or getattr(model, "forward_std", None)
    if vfunc is None:
        raise AttributeError("No forward function found in model")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            with timer:
                outputs = vfunc(imgs)
            _preds = F.softmax(outputs, 1).argmax(1)
            correct += torch.sum(_preds == labels)
            preds["preds"].append(_preds.detach().cpu().numpy())
            preds["labels"].append(labels.detach().cpu().numpy())
            total += labels.shape[0]
            if i % 5 == 4:
                print(f"{(i / len(val_loader)) * 100} percent done in {timer.time} seconds")
                print(f"Correct: {correct}, Total: {total}")
                timer.clear()
    return correct, total, preds


def finetune(model, model_name, dataloaders, num_epochs=10, lr=2e-04,
             criterion=nn.CrossEntropyLoss(), gpu=0):
    if gpu is not None:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    learning_rate = lr
    total_step = len(dataloaders["train"])
    trainable_params = [x for x in model.parameters() if x.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    # print("Validating small sample for model before training.")
    # validate_subsample(model, dataloaders["val"], gpu)
    timer = Timer()
    epoch_timer = Timer(True)
    loop_timer = Timer()
    total_loss = 0
    for epoch in range(num_epochs):
        model = model.train()
        correct = 0
        total = 0
        with epoch_timer:
            for i, batch in enumerate(dataloaders["train"]):
                with timer:
                    images, labels = batch
                with loop_timer:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    correct += torch.sum(outputs.detach().argmax(1) == labels)
                    total += len(labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if (i+1) % 10 == 0:
                    print(f"Epoch {epoch+1}, iteration {i+1}/{total_step}," +
                          f" correct {correct}/{total}",
                          f" average loss per batch {total_loss / 10}" +
                          f" in time {loop_timer.time}")
                    total_loss = 0
                    loop_timer.clear()
                i += 1
        print(f"Trained one epoch on device {device} in {epoch_timer.time} seconds")
        print(f"Correct {correct}/{total} for epoch {epoch}")
        print("Validating model")
        correct, total, _ = validate(model, dataloaders["val"], gpu)
        print(f"Correct {correct}/{total}")
        epoch_timer.clear()


def validate_with_selectivity(model, val_loader, num_classes, hidden_outputs, gpu=0):
    timer = Timer(True)
    model = model.eval()
    if util.have_cuda():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    correct = 0
    total = 0
    preds = {"preds": [], "labels": []}
    vfunc = getattr(model, "forward", None) or getattr(model, "forward_std", None)
    if vfunc is None:
        raise AttributeError("No forward function found in model")

    # define activations
    temp = [None for _ in range(num_classes)]
    activations = dict(layer1=temp.copy(), layer2=temp.copy(), layer3=temp.copy(),
                       layer4=temp.copy())

    if not hasattr(model, "layer4"):
        activations.pop("layer4")
    if not hasattr(model, "avgpool") and "avgpool" in activations:
        activations.pop("avgpool")

    with torch.no_grad():
        num_labels = torch.zeros(num_classes, dtype=torch.long)
        if util.have_cuda():
            num_labels = num_labels.cuda(gpu)
        for i, batch in enumerate(val_loader):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            with timer:
                outputs = vfunc(imgs)
            prepare_activations(activations, hidden_outputs, labels, num_classes)
            num_labels += torch.bincount(labels, minlength=num_classes)
            _preds = F.softmax(outputs, 1).argmax(1)
            correct += torch.sum(_preds == labels)
            preds["preds"].append(_preds.detach().cpu().numpy())
            preds["labels"].append(labels.detach().cpu().numpy())
            total += labels.shape[0]
            if i % 5 == 4:
                print(f"{(i / len(val_loader)) * 100} percent done in {timer.time} seconds")
                print(f"Correct: {correct}, Total: {total}")
                timer.clear()
    return correct, total, activations, preds


def finetune_with_selectivity_regularizer(model, model_name, dataloaders,
                                          num_classes, alpha,
                                          num_epochs=10, lr=2e-04,
                                          criterion=nn.CrossEntropyLoss(),
                                          print_frequency=10,
                                          gpu=0):
    if gpu is not None:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # Register hook and account for variations in num_layers
    temp = [None for _ in range(num_classes)]
    activations = dict(layer1=temp.copy(), layer2=temp.copy(), layer3=temp.copy(),
                       layer4=temp.copy())
    hidden_outputs: Dict[str, torch.Tensor] = {}
    register_hook(model, activations, hidden_outputs)
    if not hasattr(model, "layer4"):
        activations.pop("layer4")
    if not hasattr(model, "avgpool") and "avgpool" in activations:
        activations.pop("avgpool")

    def reset_activations(activations):
        activations = dict(layer1=temp.copy(), layer2=temp.copy(), layer3=temp.copy(),
                           layer4=temp.copy())
        if not hasattr(model, "layer4"):
            activations.pop("layer4")
        if not hasattr(model, "avgpool") and "avgpool" in activations:
            activations.pop("avgpool")
        return activations

    learning_rate = lr
    total_step = len(dataloaders["train"])
    trainable_params = [x for x in model.parameters() if x.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    correct, total, activations, _ = validate_with_selectivity(model, dataloaders["val"], num_classes,
                                                               hidden_outputs, gpu)
    selectivities: Dict[str, List] = {x: [] for x in activations.keys()}
    with torch.no_grad():
        selectivity = calculate_selectivity_torch(activations)
        for layer in selectivity:
            selectivities[layer].append(torch.mean(torch.tensor([x[1] for x in selectivity[layer]])).item())
    print(f"Selectivity before training: {[selectivities[layer][-1] for layer in selectivities]}")
    timer = Timer()
    epoch_timer = Timer(True)
    loop_timer = Timer()
    total_loss = 0
    for epoch in range(num_epochs):
        model = model.train()
        correct = 0
        total = 0
        with epoch_timer:
            for i, batch in enumerate(dataloaders["train"]):
                images, labels = batch
                with loop_timer:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    with timer:
                        activations = reset_activations(activations)
                        prepare_activations(activations, hidden_outputs, labels, num_classes)
                        num_labels = torch.bincount(labels, minlength=num_classes)
                        mean_selectivity = selectivity_regularizer(activations, num_labels)
                    print(f"Selectivity time {timer.time}")
                    # loss = criterion(outputs, labels)
                    loss = criterion(outputs, labels) + alpha * (sum(mean_selectivity.values()) / len(activations))
                    correct += torch.sum(outputs.detach().argmax(1) == labels)
                    total += len(labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if not (i+1) % print_frequency:
                    print(f"Epoch {epoch+1}, iteration {i+1}/{total_step}," +
                          f" correct {correct}/{total}",
                          f" average loss per batch {total_loss / 10}" +
                          f" in time {loop_timer.time}")
                    total_loss = 0
                    loop_timer.clear()
                i += 1
        print(f"Trained one epoch on device {device} in {epoch_timer.time} seconds")
        print(f"Correct {correct}/{total} for epoch {epoch}")
        print("Validating model")
        # correct, total, _ = validate(model, dataloaders["val"], gpu)
        correct, total, activations, _ = validate_with_selectivity(model, dataloaders["val"], num_classes,
                                                                   hidden_outputs, gpu)
        with torch.no_grad():
            selectivity_delta = {}
            selectivity = calculate_selectivity_torch(activations)
            for layer in selectivity:
                selectivities[layer].append(torch.mean(torch.tensor([x[1] for x in selectivity[layer]])).item())
                selectivity_delta[layer] = selectivities[layer][-1] - selectivities[layer][-2]
        print(f"Selectivity after epoch {epoch}: {[selectivities[layer][-1] for layer in selectivities]}")
        print(f"Change in selectivity after epoch {epoch}: "
              f"{selectivity_delta}, {sum(selectivity_delta.values())}")
        print(f"Correct {correct}/{total}")
        epoch_timer.clear()
    torch.save({"state_dict": model.state_dict(),
                "selectivities": selectivities},
               f"{model_name}-selectivity.pth")
