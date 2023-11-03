import torch
from torch import nn

from engine import *
from helper_functions import *
from ViT import *
from main import *


device = 'mps'

image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                           destination="pizza_steak_sushi")

train_dir = image_path / "train"
test_dir = image_path / "test"

data_loaders()

vit = ViT(num_classes=3)

optimizer = torch.optim.Adam(params=vit.parameters(),
                             lr=3e-3, 
                             betas=(0.9, 0.999),    
                             weight_decay=0.3) 


loss_fn = torch.nn.CrossEntropyLoss()

set_seeds(42)

results = engine.train(model=vit,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=10,
                       device=device)

