import pytorch
from pytorch import nn


class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 embedding_dim: int = 768):
        super().__init__()

        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 patch_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        self.flatten = nn.Flatten(start_dim=2,
                                  end_dim=3)


    def forward(self, x):

        image_resolution = x.shape[-1]
        assert image_resolution % patch_size == 0, f"Image resolution {image_resolution} must be divisible by patch size {patch_size}"

        x_pathed = self.patcher(x)
        x_flattened = self.flatten(x_pathed)

        return x_flattened.permute(0, 2, 1)
