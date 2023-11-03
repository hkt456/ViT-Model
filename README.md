# ViT-Model

A reimplementation of the ViT visual model based on the architecture of a transformer orginally designed for text-base tasks.

> This is my implementation of the ViT Model for practicing Pytorch

<img src = "./img/vit.gif" width = 500px> </img>

# Background

ViT is a computer vision model that is built on the attention mechanism and the well-known architecture of Transformer to ensure the use of
contextual information (including the position of each frame and the labels granted for the image in this case for us).

The following is the research paper: <a href="https://openreview.net/pdf?id=YicbFdNTTy">Research Paper</a>

The official Jax repository is <a href="https://github.com/google-research/vision_transformer">here</a>.

A tensorflow2 translation also exists <a href="https://github.com/taki0112/vit-tensorflow">here</a>, created by research scientist <a href="https://github.com/taki0112">Junho Kim</a>! 🙏

# Structure of Source

```bash
.
├── ViT.py
├── __pycache__
│   ├── ViT.cpython-310.pyc
│   ├── data_setup.cpython-310.pyc
│   ├── data_setup.cpython-38.pyc
│   ├── engine.cpython-310.pyc
│   ├── engine.cpython-38.pyc
│   ├── helper_functions.cpython-310.pyc
│   ├── helper_functions.cpython-38.pyc
│   ├── main.cpython-310.pyc
│   ├── mlp.cpython-310.pyc
│   ├── msa.cpython-310.pyc
│   ├── patch_embedding.cpython-310.pyc
│   ├── path_embedding.cpython-310.pyc
│   └── transformer_encoder.cpython-310.pyc
├── data_setup.py
├── engine.py
├── helper_functions.py
├── main.py
├── mlp.py
├── msa.py
├── patch_embedding.py
├── train.py
└── transformer_encoder.py
```

# Installation

```bash
$ git clone https://github.com/hkt456/ViT-Model.git
$ git cd ViT-Model
```

# Usage

In order to get the overview of the structure of the Multihead Attention layer, Multi-layer Perceptron layer, Transformer Encoder, and the
ViT model:

```bash
python3 source/main.py
```

For training and testing out the model, you can use data_setup for downloading the neccessary data and set up dataloaders:

```python

from data_setup import *
get_data() # Automatically downloads a sample classification image datasets
create_dataloaders() # Returns a tuple of (train_dataloader, test_dataloader, class_names) where class_names is a list of the target classes.

```

Despite not having set up automatic training, there's already a template for training, testing, and evaluating the performance of the model:

```python
from engine import *
train()
 """
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]}
    For example if training for epochs=2:
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]}
"""
```

There are also functions for illustrating accuravy, loss,... a lot of things. Feel free to check out `helper_functions.py`
