import torch
import torch.nn as nn
from itertools import product
import numpy as np

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models

import sys

sys.path.append("../model")

from .downsample import Downsample


def conv3x3(in_features, out_features, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_features, out_features, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_features, out_features, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_features, out_features, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        # self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FeatureExtractor(nn.Module):
    def __init__(self):

        self.inplanes = 64
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Sequential(
            *[
                nn.MaxPool2d(kernel_size=2, stride=1),
                Downsample(filt_size=3, stride=2, channels=64),
            ]
        )
        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)

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

    # STILL NO IDEA
    def _make_layer(self, block, planes, num_block, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = (
                [Downsample(filt_size=3, stride=stride, channels=self.inplanes),]
                if (stride != 1)
                else []
            )
            downsample += [
                conv1x1(self.inplanes, planes * block.expansion, 1),
                nn.BatchNorm2d(planes * block.expansion),
            ]
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_block):
            layers.append(block(self.inplanes, planes, 1, None))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# compare features in square neighborhood
class CorrNeigh(nn.Module):
    def __init__(self, kernelSize):
        super(CorrNeigh, self).__init__()
        assert kernelSize % 2 == 1
        self.kernelSize = kernelSize
        self.paddingSize = kernelSize // 2
        self.padding = torch.nn.ZeroPad2d(self.paddingSize)

    def forward(self, x, y):

        ## x, y should be normalized
        w, h = x.size()[2:]
        coef = []
        y = self.padding(y)
        ## coef is the feature similarity between (i,j) and (i-r, j-r) with -kernel < r < +kernel
        for i, j in product(range(self.kernelSize), range(self.kernelSize)):
            coef.append(
                torch.sum(x * y.narrow(2, i, w).narrow(3, j, h), dim=1, keepdim=True)
            )  # after sum: size = (n , 1 , w , h)
        coef = torch.cat(coef, dim=1)  # size (n, kernelSize**2, w, h)

        return coef


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
            self.conv4 = conv3x3(128, kernelSize * kernelSize)
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

        if self.network == 'netFlowCoarse':
            self.softmax = torch.nn.Softmax(dim=1)
        elif self.network == 'netMatch':
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
