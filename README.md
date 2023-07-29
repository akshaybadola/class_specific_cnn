# README

- This is the code for the paper [Decomposing the Deep](https://arxiv.org/abs/2112.07719), published
  as https://link.springer.com/article/10.1007/s00521-023-08441-z.
- This is an updated version of https://github.com/akshaybadola/decomposing-the-deep/ with cleaner
  code. Still quite messy.
- The python requirements are given in requirements.txt.
- You'll need the CIFAR-10 and/or Imagenet data present in a `data` folder to run the experiments
- The *influential filters* can vary according to the trained instance of the model.
  We provide the filters for CIFAR-10 on the checkpoint given in file `resnet20-12fca82f.th`
- There are also saved `decomposed` after fine-tuning with our method.
  And `regularized` weights after applying selectivity regularizer as given by Leavitt et al. in
  Selectivity Considered Harmful (https://openreview.net/forum?id=8nl0k08uMi).

## Usage

- For usage see `main.py`

### Finetuning after decomposition on pretrained resnet20

The weights are staged in the repo. This will evaluate the model first print
the accuracy before fine-tuning and then do fine tuning after decomposition to
demonstrate that original accuracy can be restored. Resnet20 with CIFAR-10
happens fairly quickly. Resnet50 with Imagenet takes a few epochs.

`python main.py finetune --model resnet20 --weights resnet20-12fca82f.th --finetune-method decomposed -k 5 -b 256`

Command line example for Resnet50 + Imagenet:

`python main.py finetune --model resnet50 --finetune-method decomposed -k 64 -b 64`

You don't need to specify weights file for Resnet50 as it'll use pytorch pretrained model.


### Retrieving *influential indices*

If `indices` file is not given, then the code will automatically find the selective
indices from the final layer. To get the indices *for any layer*, you'll have to
give the name of the submodule in pytorch. You can do a `[x[0] for x in model.named_modules()]`
at a python/IPython prompt after loading the model to see the names of submodules.

Example to retrieve from layer2 -> conv2 -> ReLU, that is, after ReLU of second conv module
from "layer" 2.

`python main.py get_inds --model resnet20 --weights resnet20-12fca82f.th --layer-name layer2.1.relu2 -k 5 -b 256 -g 0`

This will store the indices in a file `indices_{layer2.1.relu2}.pkl`

### Calculating Selectivity \mu

The functions are there in `main.py`. Will write example later.

**TODO**

### Calculating Selectivity \psi

The functions are there in `main.py`. Will write example later.

**TODO**


## Citation

If you use any part of this code, please consider citing:

```
@article{badola2021decomposing,
  url={https://doi.org/10.1007/s00521-023-08441-z},
  doi={10.1007/s00521-023-08441-z},
  year={2023},
  month={June},
  voluem={35},
  issue={18},
  pages={13583-13596},
  journal={Neural computing & applications},
  author={Badola, Akshay and Roy, Cherian and Padmanabhan, V. and Lal, R.},
  title={Decomposing the deep: finding class-specific filters in deep CNNs},
}
```
