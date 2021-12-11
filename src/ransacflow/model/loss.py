import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from kornia.losses import ssim_loss


class MaskedSSIMLoss(nn.Module):
    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = window_size
        # TODO create margin and mask the input

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return ssim_loss(img1, img2, window_size=self.window_size)


class MatchabilityLoss(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class ReconstructionLoss(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class CycleConsistencyLoss(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class PerceptualLoss(nn.Module):
    """
    A type of content loss introduced in the "Perceptual Losses for Real-Time Style
    "Transfer and Super-Resolution framework.

    In this implementatoin, the perceptual loss is based on the ReLU activation layers
    of a pre-trained 16 layer VGG network.

    Reference:
        - PyTorch implementation of VGG perceptual loss
          https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
        - Probing shallower: Perceptual loss trained phase extraction neural network
          for artifact-free reconstruciton at low photon budget
          https://doi.org/10.1364/OE.381301
    """

    def __init__(self):
        super().__init__()

        pretrained = torchvision.models.vgg16(pretrained=True)
        pretrained.eval()

        # extract blocks of layers
        blocks = [
            pretrained.features[:4],
            pretrained.features[4:9],
            pretrained.features[9:16],
            pretrained.features[16:23],
        ]
        # disable gradient computations only, to allow possible back propagation
        for block in blocks:
            for param in block.parameters():
                param.requires_grad = False
        # we don't need forward, but we do want to keep track of parameters
        self.blocks = nn.ModuleList(blocks)

        self.normalize = (
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def forward(self, sample, target, feature_layers=(0, 1, 2, 3)):
        sample = self.normalize(sample)
        target = self.normalize(target)

        loss = 0
        x, y = sample, target
        for i, block in enumerate(self.blocks):
            x, y = block(x), block(y)
            if i in feature_layers:
                loss += nn.functional.l1_loss(x, y)
        return loss
