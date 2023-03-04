import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common_pyutil.monitor import Timer


def conv_or_linear(x):
    return isinstance(x, (torch.nn.Conv2d, torch.nn.Linear))


def depth(x, pred):
    children = [*x.children()]
    if not children:
        if pred(x):
            return 1
        else:
            return 0
    else:
        return sum([depth(c, pred) for c in children])


def json_defaults(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy().tolist()
    else:
        return x


def load_fixing_names(model, weights_or_state_dict):
    model_keys = [*model.state_dict().keys()]
    if "state_dict" in weights_or_state_dict:
        state_dict = weights_or_state_dict["state_dict"]
    keys = [*state_dict.keys()]
    for k in keys:
        if k in model_keys:
            continue
        if k.replace("module.", "") in model_keys:
            state_dict[k.replace("module.", "")] = state_dict.pop(k)
    return model.load_state_dict(state_dict, strict=False)


def have_cuda():
    return torch.cuda.is_available()


def validate(model, val_loader, gpu=0, print_only=False):
    timer = Timer(True)
    model = model.eval()
    if have_cuda():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    correct = 0
    total = 0
    preds = {"preds": [], "labels": []}
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            with timer:
                outputs = model(imgs)
            _preds = F.softmax(outputs, 1).argmax(1)
            correct += torch.sum(_preds == labels)
            preds["preds"].append(_preds.detach().cpu().numpy())
            preds["labels"].append(labels.detach().cpu().numpy())
            total += labels.shape[0]
            if i % 5 == 4:
                print(f"{(i / len(val_loader)) * 100} percent done in {timer.time} seconds")
                print(f"Correct: {correct}, Total: {total}")
                timer.clear()
    if print_only:
        print(f"correct: {correct}, total: {total}, accuracy: {correct/total*100}")
    else:
        return correct, total, preds
