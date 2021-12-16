import itertools

import antialiased_cnns
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FeatureExtractor", "NeighborCorrelator"]


class FeatureExtractor(nn.Module):
    """
    TBD

    Args:
        pretrained (bool, optional): Using pretrained model.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        resnet = antialiased_cnns.resnet18(pretrained=pretrained)

        # the original work replace the first conv layer with
        #   - smaller kernel
        #   - stride=1
        #   - no bias
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        layer_list = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3"]
        layers = [getattr(resnet, name) for name in layer_list]
        resnet = nn.Sequential(*layers)

        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)


class NeighborCorrelator(nn.Module):
    """
    Calculate cosine similarity over a sliding window.

    Args:
        kernel_size (int): Patch of neighbors to compare with.
    """

    def __init__(self, kernel_size: int):
        super().__init__()

        assert kernel_size % 2 == 1, "kernel size has to be odd"
        self.kernel_size = kernel_size

        self.zero_padding = nn.ZeroPad2d(kernel_size // 2)

    def forward(self, x, y):

        # normalize vectors first, so we don't have to divide them later
        x = F.normalize(x)
        y = F.normalize(y)

        h, w = x.shape[-2:]
        # move along the other image
        y = self.zero_padding(y)

        similarity = []
        for i, j in itertools.product(range(self.kernel_size), range(self.kernel_size)):
            similarity.append(
                # (B, 1, W, H)
                torch.sum(x * y[..., i : i + w, j : j + h], dim=1, keepdim=True)
            )

        # (B, K*K, W, H)
        similarity = torch.cat(similarity, dim=1)

        return similarity
