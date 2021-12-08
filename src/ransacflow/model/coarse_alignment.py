import kornia as K
import numpy as np
import torch
import torch.nn as nn
from ransacflow.util import get_model_root
from torchvision import models, transforms


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
    ws = torch.arange(0, nw, device=feature.device)
    hs = torch.arange(0, nh, device=feature.device)
    w, h = torch.meshgrid(ws, hs, indexing="xy")

    # shift the grid so they are centered on pixels
    w = (w + 0.5) / nw
    h = (h + 0.5) / nh

    return (h, w)


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
        threshold: float = 0.05,
        max_iter: int = 10000,
        use_moco: bool = False,
    ):
        super().__init__()

        # save parameters
        if not n_scales % 2:
            raise ValueError("'n_scales' has to be odd")
        if scale_ratio <= 1:
            raise ValueError("'scale_ratio' has to be >= 1")

        # preprocess routines
#         self.resnet_normalize = transforms.Normalize(
#             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#         )

        # load resnet50
        if not use_moco:
            resnet = models.resnet50()
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

        self.ransac = K.geometry.RANSAC(
            model_type="homography",  # we only work with homography
            inl_th=threshold,
            max_iter=max_iter,
        )

    def forward(self, I_src: torch.Tensor, I_dst: torch.Tensor):
        """
        TBD

        Assume input iamges are properly normalized for ResNet.

        Args:
            I_src (Tensor):
            I_dst (Tensor):
        """
        feature_dst, (h_dst, w_dst) = self._extract_feature(I_dst)
        feature_src, (h_src, w_src) = self._extract_multiscale_feature(I_src)

        _, matches = K.feature.match_mnn(feature_src, feature_dst)

        kp_src = torch.stack([h_src[matches[:, 0]], w_src[matches[:, 0]]], dim=-1)
        kp_dst = torch.stack([h_dst[matches[:, 1]], w_dst[matches[:, 1]]], dim=-1)

        model, _ = self.ransac(kp_src, kp_dst)

        # warp the result
        warper = K.geometry.HomographyWarper(*I_src.shape[-2:])
        I_src_warp = warper(I_src.unsqueeze(0), model.unsqueeze(0))

        return I_src_warp

    def _extract_feature(self, image: torch.Tensor):
        """Extract feature map for a given image.

        Args:
            image (Tensor): [description]

        Returns:
            TBD
        """
        image = self.resnet_normalize(image)

        image = image.unsqueeze(0)  # [C, H, W] -> [B, C, H, W]
        feature = self.resnet(image)
        feature = nn.functional.normalize(feature)

        h, w = create_grid(feature)
        h = torch.flatten(h)
        w = torch.flatten(w)

        feature = torch.flatten(feature, start_dim=-2)
        feature = feature.squeeze().t()  # [1, D, N] -> [N, D]

        return feature, (h, w)

    def _extract_multiscale_feature(self, image: torch.Tensor):
        """
        Build feature map within predefined multiple scales.

        Args:
            image (Tensor):

        Returns:

        """
        feature_list = []
        h_list = []
        w_list = []
        for scale in self.scale_list:
            image_scaled = K.geometry.transform.rescale(
                image, scale, align_corners=False
            )

            feature, (h, w) = self._extract_feature(image_scaled)
            feature_list.append(feature)
            h_list.append(h)
            w_list.append(w)

        # concat results from every scale as if finding additional keypoints
        feature_list = torch.vstack(feature_list)
        h_list = torch.hstack(h_list)
        w_list = torch.hstack(w_list)

        return feature_list, (h_list, w_list)
