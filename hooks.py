from typing import List, Dict
from functools import partial
import torch
import torch.nn.functional as F

from util import have_cuda


class FilterOutputs:
    def __init__(self, model, module, persistent=False, debug=False):
        self.model = model
        self.model = self.model.eval()
        self.result = []
        self.persistent = persistent

        def _debug_hook(result, *x):
            import ipdb; ipdb.set_trace()
            result.append(x)

        debug_hook = partial(_debug_hook, self.result)
        if isinstance(module, str):
            self.mod = getattr(model, module)
        if debug:
            self.handle = self.mod.register_forward_hook(debug_hook)
        else:
            self.handle = self.mod.register_forward_hook(lambda *x: self.result.append(x))

    def __call__(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            self.model(x)

    @property
    def results(self):
        mod, inputs, outputs = self.result[-1]
        return {"module": mod, "input": inputs, "output": outputs}


def model_set_labels(model: torch.nn.Module, layers: List[str],
                     labels: torch.Tensor):
    model.labels = labels.cpu().numpy()
    for i, layer in enumerate(layers):
        for j, child in enumerate(getattr(model, layer).children()):
            child.labels = model.labels


def get_keys(model, layers, layer_only=False):
    keys = [(0, 0, 0)]
    for i, layer in enumerate(layers):
        for j, child in enumerate(getattr(model, layer).children()):
            for k in range(1, 10):
                if getattr(child, f"conv{k}", None):
                    keys.append((i+1, j, k))
    if layer_only:
        keys = [*set(k[:2] for k in keys)]
        keys.sort()
    return keys


def check_track_for(track_for, keys):
    if isinstance(track_for, int):
        track_for = [keys[track_for]]
    elif isinstance(track_for, tuple):
        if track_for in keys:
            track_for = [track_for]
        else:
            raise ValueError(f"Invalid layer to track {track_for}")
    elif isinstance(track_for, list):
        _track_for = []
        for t in track_for:
            if isinstance(t, int):
                _track_for.append(keys[t])
            elif isinstance(t, tuple) and t in keys:
                _track_for.append(t)
            else:
                raise ValueError(f"Unknown type to track {t}")
        track_for = _track_for
    else:
        raise ValueError(f"Unknown type of track_for {track_for}")
    return track_for


def _debug_hook(results, t, *x):
    import ipdb; ipdb.set_trace()
    results[t].append(x[-1])


def get_outputs_at(model, model_name, dataloaders, track_for, gpu=0):
    if have_cuda:
        model = model.cuda(gpu)
    if model_name == "resnet20":
        layers = ["layer1", "layer2", "layer3"]
    elif "resnet" in model_name.lower():
        layers = ["layer1", "layer2", "layer3", "layer4"]
    else:
        raise ValueError(f"Unknown model {model_name}")
    keys = get_keys(model, layers, layer_only=True)
    model = model.eval()
    print(f"Will track for layers {track_for}")
    total_iters = len(dataloaders["train"])
    track_for = check_track_for(track_for, keys)
    results: Dict[tuple, List] = {t: [] for t in track_for}
    labels = []
    # results = []
    for t in track_for:
        mod = model.get_child(t)
        # only get output
        mod.register_forward_hook(lambda *x: results[t].append(x[-1].detach().cpu()))
        # debug_hook = partial(_debug_hook, results, t)
        # mod.register_forward_hook(debug_hook)
    with torch.no_grad():
        for i, batch in enumerate(dataloaders["train"]):
            imgs, _labels = batch
            if have_cuda:
                imgs = imgs.cuda(gpu)
            labels.append(_labels.cpu().numpy())
            _ = model.forward(imgs)
            if (i + 1) % 100 == 0:
                print(f"Done {i+1} iterations out of {total_iters}")
    return results, labels
