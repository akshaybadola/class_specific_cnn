# README

- This is the code for the paper [Decomposing the Deep](https://arxiv.org/abs/2112.07719), published
  as https://link.springer.com/article/10.1007/s00521-023-08441-z.
- This is an updated version of https://github.com/akshaybadola/decomposing-the-deep/ with cleaner
  code. Still quite messy.
- The python requirements are given in requirements.txt.
- You'll need the CIFAR-10 and/or Imagenet data present in a `data` folder to run the experiments
- The *influential filters* can vary according to the trained instance of the model.
  We provide the filters for CIFAR-10 on the checkpoint given in file `resnet20-12fca82f.th`
- There are also saved `decomposed` and `regularized` weights as given by Leavitt et al. in
  https://openreview.net/forum?id=8nl0k08uMi

## Usage

- For usage see `main.py`. I'll add rest of documentation soon.
