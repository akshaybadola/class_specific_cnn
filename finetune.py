import torch
import torch.nn as nn
import torch.nn.functional as F

from common_pyutil.monitor import Timer

import util


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


def finetune(model, model_name, dataloaders, num_epochs=10, lr=2e-04, gpu=0):
    if gpu is not None:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    learning_rate = lr
    criterion = nn.CrossEntropyLoss()
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
