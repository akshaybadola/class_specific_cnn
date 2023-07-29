import numpy as np
import torch
import torch.nn as nn


have_cuda = torch.cuda.is_available()


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
    elif "model_state_dict" in weights_or_state_dict:
        state_dict = weights_or_state_dict["state_dict"]
    else:
        state_dict = weights_or_state_dict
    keys = [*state_dict.keys()]
    for k in keys:
        if k in model_keys:
            continue
        if k.replace("module.", "") in model_keys:
            state_dict[k.replace("module.", "")] = state_dict.pop(k)
    return model.load_state_dict(state_dict, strict=False)
