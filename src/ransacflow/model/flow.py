import torch
import torch.nn as nn

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
    """

    All output from child classes are downsized by the feature extractor, we need to
    upsample it back to original size. This operation is left for user, since data pass
    to this network is already in feature space. One can upsample by
        flow = F.interpolate(flow, size=image_size, mode='bilinear')

    Args:
        kernel_size (TBD): TBD
    """

    def __init__(self, kernel_size: int):
        super().__init__()

        self.kernel_size = kernel_size

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = conv2d_3x3(kernel_size * kernel_size, 512)
        self.bn1 = nn.BatchNorm2d(512)

        self.conv2 = conv2d_3x3(512, 256)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = conv2d_3x3(256, 128)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # this output is simply the CNN result, need to add an activation layer by
        # child classes to make it usable
        return x


class FlowPredictor(BaseFlowPredictor):
    """
    TBD

    Output of the flow prediction is within [-1, 1], which is a relative length in the
    K-by-K window. Need to scale it to be proportional to image size, such as
        ny, nx = image_size
        flow_x /= nx / 2.0
        flow_y /= ny / 2.0

    Args:
        kernel_size (TBD): TBD
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        n_features = self.conv3.out_channels
        self.conv4 = conv2d_3x3(n_features, self.kernel_size * self.kernel_size)

        self.softmax = nn.Softmax(dim=1)

        # generate the for local coordinate [-K//2, K//2]
        ks = torch.arange(0, self.kernel_size) - self.kernel_size // 2
        grid_x, grid_y = torch.meshgrid(ks, ks, indexing="xy")

        # save these grid coordinates, and flatten it along the way
        self.register_buffer("grid_x", grid_x.reshape(1, -1, 1, 1), persistent=False)
        self.register_buffer("grid_y", grid_y.reshape(1, -1, 1, 1), persistent=False)

    def forward(self, x):
        x = super().forward(x)

        x = self.conv4(x)
        x = self.softmax(x)

        # transform x from patches of local coordinate shifts, into global coordinate
        flow_x = torch.sum(x * self.grid_x, dim=1, keepdim=True)
        flow_y = torch.sum(x * self.grid_y, dim=1, keepdim=True)
        # combine both flow to yield the final output
        x = torch.cat([flow_x, flow_y], dim=1)

        return x


class MatchabilityPredictor(BaseFlowPredictor):
    """
    TBD

    Args:
        kernel_size (TBD): TBD
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        n_features = self.conv3.out_channels
        self.conv4 = conv2d_3x3(n_features, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = super().forward(x)

        x = self.conv4(x)
        x = self.sigmoid(x)

        return x
