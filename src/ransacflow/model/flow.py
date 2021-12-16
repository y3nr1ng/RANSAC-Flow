from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FlowPredictor", "MatchabilityPredictor"]


def conv2d_3x3(in_channels: int, out_channels: int):
    """
    We are reusing this 3x3 Conv2d pattern a lot. Save it as a function.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
    """
    # We use BatchNorm2d immediately affter Conv2d, which will remove the channel mean.
    # There is no point of adding bias to Conv2d.
    #   https://github.com/kuangliu/pytorch-cifar/issues/52
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)


class BaseFlowPredictor(nn.Module):
    def __init__(self, image_size: Union[int, Tuple[int, int]], kernel_size: int):
        super().__init__()

        assert isinstance(image_size, (int, tuple)), "unknown image size data type"
        if isinstance(image_size, int):
            # (h, w) = (s, s)
            image_size = (image_size, image_size)
        else:
            # it is a tuple, (h, w)
            assert len(image_size) == 2, "len(image_size) should equal to 2"
        self.image_size = image_size

        assert kernel_size % 2 == 1, "kernel size has to be odd"
        self.kernel_size = kernel_size

        self.conv1 = conv2d_3x3(kernel_size * kernel_size, 512)
        self.bn1 = nn.BatchNorm2d(512)

        self.conv2 = conv2d_3x3(512, 256)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = conv2d_3x3(256, 128)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self._forward(x)

        # since all child class will need upsample later on, we do it here
        x = F.interpolate(x, size=self.image_size, mode="bilinear")

        return x

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)

        return x


class FlowPredictor(BaseFlowPredictor):
    """
    TBD

    Args:
        TBD
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        n_features = self.conv3.out_channels
        self.conv4 = conv2d_3x3(n_features, self.kernel_size * self.kernel_size)

        self.softmax = nn.Softmax(dim=1)

        # generate the for local coordinate [-K//2, K//2]
        ks = torch.arange(0, self.kernel_size) - self.kernel_size // 2
        grid_x, grid_y = torch.meshgrid(ks, ks, indexing="xy")
        # scale the shift (grid) vector(s) so that it is proportional to the image size
        # NOTE
        #   expand() in meshgrid() only creates new view, need explicit assignment to
        #   allocate the full memory
        ny, nx = self.image_size
        grid_x = grid_x / (nx / 2.0)
        grid_y = grid_y / (ny / 2.0)

        # save these grid coordinates, and flatten it along the way
        self.register_buffer("grid_x", grid_x.view(1, -1, 1, 1))
        self.register_buffer("grid_y", grid_y.view(1, -1, 1, 1))

    def _forward(self, x):
        x = super()._forward(x)

        x = self.conv4(x)
        x = self.softmax(x)

        # transform x from patches of local coordinate shifts, into global coordinate
        flow_x = torch.tensordot(x, self.grid_x, dims=1)
        flow_y = torch.tensordot(x, self.grid_y, dims=1)
        # combine both flow to yield the final output
        x = torch.cat([flow_x, flow_y], dim=1)

        return x


class MatchabilityPredictor(BaseFlowPredictor):
    """
    TBD

    Args:
        TBD
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        n_features = self.conv3.out_channels
        self.conv4 = conv2d_3x3(n_features, 1)

        self.sigmoid = nn.Sigmoid()

    def _forward(self):
        x = super()._formward()

        x = self.conv4(x)
        x = self.sigmoid(x)

        return x
