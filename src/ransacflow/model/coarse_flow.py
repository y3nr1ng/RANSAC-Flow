import antialiased_cnns
import torch.nn as nn

__all__ = ["FeatureExtractor"]


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
