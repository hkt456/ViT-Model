from data_setup import *
from engine import *
from helper_functions import *
from path_embedding import *
from msa import *
from transformer_encoder import *


import torch
from torch import nn
from torchvision import transforms

# Download pizza, steak, sushi images from GitHub
image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                           destination="pizza_steak_sushi")

train_dir = image_path / "train"
test_dir = image_path / "test"

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
PATCH_SIZE = 16


# Create data loaders
def data_loaders():
    manual_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    print(f"Manually created transforms: {manual_transforms}")

    train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                        test_dir=test_dir,
                                                                        transform=manual_transforms,
                                                                        batch_size=BATCH_SIZE,)
    print(train_dataloader)
    print(test_dataloader)
    print(class_names)


def get_summary_transformer_encoder():
    transformer_encoder = TransformerEncoderBlock()
    summary(model = transformer_encoder,
        input_size=(1, 768, 768),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])


# # Get a summary of the input and outputs of PatchEmbedding (uncomment for full output)
def get_summary_patchEmbedding():
    random_input_image = (1, 3, 224, 224)
    summary(PatchEmbedding(),
        input_size=random_input_image, # try swapping this for "random_input_image_error"
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])


def __main__():
    
    data_loaders()
    get_summary_patchEmbedding() 
    get_summary_transformer_encoder()



if __name__ == "__main__":
    __main__()
