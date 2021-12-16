from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_features, out_features, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_features, out_features, kernel_size=3, stride=stride, padding=1, bias=False
    )

# Net of flow and matchability
class NetFlow(nn.Module):
    def __init__(self, kernelSize, network):
        super(NetFlow, self).__init__()
        assert kernelSize % 2 == 1

        self.conv1 = conv3x3(kernelSize * kernelSize, 512)

        self.bn1 = nn.BatchNorm2d(512, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(512, 256)
        self.bn2 = nn.BatchNorm2d(256, eps=1e-05)

        self.conv3 = conv3x3(256, 128)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-05)

        self.kernelSize = kernelSize
        self.paddingSize = kernelSize // 2

        self.network = network  # network type: netFlowCoarse or netMatch

        if self.network == "netFlowCoarse":
            self.conv4 = conv3x3(128, kernelSize * kernellSize)
            self.gridY = (
                torch.arange(-self.paddingSize, self.paddingSize + 1)
                .view(1, 1, -1, 1)
                .expand(1, 1, self.kernelSize, self.kernelSize)
                .contiguous()
                .view(1, -1, 1, 1)
                .type(torch.FloatTensor)
            )
            self.gridX = (
                torch.arange(-self.paddingSize, self.paddingSize + 1)
                .view(1, 1, 1, -1)
                .expand(1, 1, self.kernelSize, self.kernelSize)
                .contiguous()
                .view(1, -1, 1, 1)
                .type(torch.FloatTensor)
            )
            # put gridX, gridY into gpu
            self.gridX, self.gridY = self.gridX.cuda(), self.gridY.cuda()
        elif self.network == "netMatch":
            self.conv4 = conv3x3(128, 1)

        if self.network == "netFlowCoarse":
            self.softmax = torch.nn.Softmax(dim=1)
        elif self.network == "netMatch":
            self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if (
                    m.in_channels != m.out_channels
                    or m.out_channels != m.groups
                    or m.bias is not None
                ):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                else:
                    print("Not initializing")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        ## make the initial matchability to 0.5
        if self.network == "netMatch":
            nn.init.normal_(self.conv4.weight, mean=0.0, std=0.0001)

    def forward(self, x, up8X=True):

        ## x, y should be normalized
        n, c, w, h = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)

        if self.network == "netFlowCoarse":
            ## scale the similarity term and do softmax (self.heat is learnt as well)
            x = self.softmax(x)

            ## flow. not sure why need to divide by h and w
            flowX = torch.sum(x * self.gridX, dim=1, keepdim=True) / h * 2
            flowY = torch.sum(x * self.gridY, dim=1, keepdim=True) / w * 2
            flow = torch.cat((flowX, flowY), dim=1)
            x = flow
        elif self.network == "netMatch":
            x = self.sigmoid(x)

        x = F.upsample_bilinear(x, size=None, scale_factor=8) if up8X else x
        return x


## Take the central part to estimate pixel transformation
def SSIM(I1Warp, I2, match, ssim):
    return ssim(I1Warp, I2, match)


def predFlowCoarse(corrKernel, NetFlowCoarse, grid, up8X=True):

    flowCoarse = NetFlowCoarse(corrKernel, up8X)  ## output is with dimension B, 2, W, H
    _, _, w, h = flowCoarse.size()
    flowGrad = flowCoarse.narrow(2, 1, w - 1).narrow(3, 1, h - 1) - flowCoarse.narrow(
        2, 0, w - 1
    ).narrow(3, 0, h - 1)
    flowGrad = torch.norm(flowGrad, dim=1, keepdim=True)
    flowCoarse = flowCoarse.permute(0, 2, 3, 1)
    flowCoarse = torch.clamp(flowCoarse + grid, min=-1, max=1)

    return flowGrad, flowCoarse


def predFlowCoarseNoGrad(corrKernel, NetFlowCoarse, grid, up8X=True):

    flowCoarse = NetFlowCoarse(corrKernel, up8X)  ## output is with dimension B, 2, W, H

    flowCoarse = flowCoarse.permute(0, 2, 3, 1)  # size: B, W, H, 2
    flowCoarse = torch.clamp(
        flowCoarse + grid, min=-1, max=1
    )  # add flowCoarse + grid( NOT image2 !)

    return flowCoarse


def predMatchability(corrKernel21, NetMatchability, up8X=True):

    matchability = NetMatchability(corrKernel21, up8X)

    return matchability
