from pathlib import Path

import kornia as K
import numpy as np
import torch
import torch.nn as nn
from ransacflow.util import get_model_root
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode


def get_resnet50_moco_state_dict() -> dict:
    """
    Get weight of ResNet50 trained with MoCo.

    Returns:
        (dict): Parameters and persistent buffers of ResNet50.
    """
    model_path = get_model_root() / "resnet50_moco.pth"
    if not model_path.exists():
        # TODO download this from remote
        raise RuntimeError(
            "Please download and put 'resnet50_moco.pth' to models folder"
        )

    # the original weight is actually a checkpoint
    #   https://github.com/bl0/moco
    # this includes hyperparameters and everything else we wants
    checkpoint = torch.load(model_path)

    # we need to drop all prefix from this checkpoint
    prefix = len("module.")
    state_dict = {k[prefix:]: v for k, v in checkpoint["model"].items()}

    return state_dict


def create_grid(feature: torch.Tensor):
    """Geenerate (W, H) grid for a tensor."""
    nh, nw = feature.shape[-2:]
    ws = torch.arange(0, nw)
    hs = torch.arange(0, nh)
    w, h = torch.meshgrid(ws, hs, indexing="xy")

    # shift the grid so they are centered on pixels
    w = (w + 0.5) / nw
    h = (h + 0.5) / nh

    return (w, h)

def mutual_matching():
    pass

class ResizeToMax(nn.Module):
    """
    Resize the input image within a maximum value.

    Args:
        max_size (int): The maximum allowd for the longer edge of the resize image.
        interpolation (InterpolationMode): Desired interpolation method.
    """

    def __init__(
        self,
        max_size: int,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()

        self.max_size = max_size
        self.interpolation = interpolation

    def forward(self, im: torch.Tensor):
        """
        Args:
            im (Tensor): Image to be scaled.
        """
        ny, nx = im.shape[-2:]  # [..., H, W]

        # determine new shape
        ratio = max(float(nx) / self.max_size, float(ny) / self.max_size)
        nx, ny = int(round(nx / ratio)), int(round(ny / ratio))

        return transforms.functional.resize(
            im, (ny, nx), interpolation=self.interpolation
        )


class CoarseAlignment(nn.Module):
    """
    TBD

    Args:
        n_scales
        scale_ratio
        max_iter
        min_size (int, optional): Minimum size of any given image dimension.
        use_moco (bool, optional): Use features learned via MoCo self-supervision.

    """

    def __init__(
        self,
        n_scales: int = 7,
        scale_ratio: float = 1.2,
        max_iter: int = 10000,
        max_size: int = 400,
        use_moco: bool = False,
    ):
        super().__init__()

        # save parameters
        if not n_scales % 2:
            raise ValueError("'n_scales' has to be odd")
        if scale_ratio <= 1:
            raise ValueError("'scale_ratio' has to be >= 1")
        self.max_iter = max_iter

        # preprocess routines
        preprocess = transforms.Compose(
            [
                ResizeToMax(max_size),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.preprocess = preprocess

        # load resnet50
        if not use_moco:
            resnet = models.resnet50(pretrained=True)
        else:
            resnet = models.resnet50()

            # moco uses a different fc layer
            #   width=1; base=64*width=64; expansion=4
            #   in_feature=base*8*expansion=[2048]
            #   out_features=low_dim=[128]
            resnet.fc = nn.Linear(2048, 128)

            state_dict = get_resnet50_moco_state_dict()
            resnet.load_state_dict(state_dict)

        # ResNet contains
        #   conv1 / bn1 / relu / maxpool / layer1-4 / avgpool / fc
        # we only want till layer3
        layer_list = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3"]
        layers = [getattr(resnet, name) for name in layer_list]
        resnet = nn.Sequential(*layers)

        resnet.eval()
        self.resnet = resnet

        # extract features at these scales
        # build from two direction, make sure we have 1x at the center
        self.scale_list = np.hstack(
            [
                np.linspace(scale_ratio, 1, n_scales // 2 + 1),
                np.linspace(1, 1 / scale_ratio, n_scales // 2 + 1)[1:],
            ]
        )

        # kornia.geometry.ransac
        #   model_type:
        #   inl_th, inlier_threshold
        #   batch_size:
        #   max_iter: max_iteration
        #   confidence
        #   max_lo_iters: max_local_iterations

        self.ransac = K.geometry.RANSAC(
            model_type="homography",  # we only work with homography
            inl_th=2,
            batch_size=2048,
            max_iter=self.max_iter,
            confidence=0.99,
        )

    def forward(self, I_src: torch.Tensor, I_dst: torch.Tensor):
        """
        TBD

        Assume input iamges are properly normalized for ResNet.

        Args:
            I_src (Tensor):
            I_dst (Tensor):
        """
        feature_dst, grid_dst = self._extract_dst_feature(I_dst)
        feature_src, grid_src = self._extract_src_multiscale_feature(I_src)

        _, matches = K.feature.match_mnn(feature_src, feature_dst)

        # TODO do ransac with matches
        kp_src, kp_dst = matches[:, :], matches[:, :]
        model, inliers = self.ransac(kp_src, kp_dst)

    def _extract_dst_feature(self, image: torch.Tensor):
        """Extract destination feature map.

        Args:
            image (Tensor): [description]

        Returns:
            TBD
        """
        image = self.preprocess(image)

        image = image.unsqueeze(0)  # [C, H, W] -> [B, C, H, W]
        feature = self.resnet(image)
        feature = nn.functional.normalize(feature)

        feature = torch.flatten(feature, start_dim=-2)
        feature = feature.t()
        grid = create_grid(feature)

        return feature, grid

    def _extract_src_multiscale_feature(self, image: torch.Tensor):
        """
        Build source feature map within multiple scales.

        Args:
            image (Tensor):

        Returns:

        """
        image = self.preprocess(image)

        image = image.unsqueeze(0)  # [C, H, W] -> [B, C, H, W]
        feature = self.resnet(image)
        feature = nn.functional.normalize(feature)

        feature = torch.flatten(feature, start_dim=-2)
        feature = feature.t()
        print(feature.shape)

        return None, None
