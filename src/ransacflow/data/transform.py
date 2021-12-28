"""
Since we are working with our own dataset, we need to patch transforms for these custom
input formats.
"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.transforms import (
    InterpolationMode,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

import logging

__all__ = [
    "RandomCropImagePair",
    "RandomHorizontalFlipImagePair",
    "ToTensorImagePair",
    "ResizeValidationImageFeaturesPair",
    "ToTensorValidationPair",
]

logger = logging.getLogger("ransacflow.data.transform")


class RandomCropImagePair(RandomCrop):
    def forward(self, im_pair):
        im0, im1 = im_pair
        assert im0.shape == im1.shape, "image pair has different dimensions"

        # the following snippet is copied from RandomCrop
        if self.padding is not None:
            im0 = F.pad(im0, self.padding, self.fill, self.padding_mode)
            im1 = F.pad(im1, self.padding, self.fill, self.padding_mode)

        width, height = F.get_image_size(im0)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            im0 = F.pad(im0, padding, self.fill, self.padding_mode)
            im1 = F.pad(im1, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            im0 = F.pad(im0, padding, self.fill, self.padding_mode)
            im1 = F.pad(im1, padding, self.fill, self.padding_mode)

        crop_dims = self.get_params(im0, self.size)

        # we need to apply the same crop location to _both_ images
        return F.crop(im0, *crop_dims), F.crop(im1, *crop_dims)


class RandomHorizontalFlipImagePair(RandomHorizontalFlip):
    def forward(self, im_pair):
        im0, im1 = im_pair
        if torch.rand(1) < self.p:
            return F.hflip(im0), F.hflip(im1)
        return im_pair


class ToTensorImagePair(ToTensor):
    def __call__(self, im_pair):
        return F.to_tensor(im_pair[0]), F.to_tensor(im_pair[1])


class ResizeValidationImageFeatures(nn.Module):
    """
    TBD

    Args:
        min_size (int): The minimum allowed for the shoerter edge of the resized image.
        interpolation (InterpolationMode, optional): Desired interpolation enum defined
            by `torchvision.transforms.InterpolationMode`.
        stride (int, optional): Image must be multiply of strides, since we downsample
            the image during feature extraction.
    """

    def __init__(
        self,
        min_size: int,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        stride: int = 16,
    ):
        super().__init__()

        self.min_size = float(min_size)
        self.interpolation = interpolation
        self.stride = stride

    def forward(self, item):
        image, features = item

        h, w = image.shape[-2:]
        h, w = float(h), float(w)

        # estimate new output size base on min size constraint
        ratio = min(h / self.min_size, w / self.min_size)
        ho, wo = round(h / ratio), round(w / ratio)

        # estimate new output size base on the stride constraint
        ho, wo = ho // self.stride * self.stride, wo // self.stride * self.stride

        # since we may round up/down in the process, recalculate final ratio to ensure
        # feature points are at correct positions
        size = (ho, wo)
        ratio_h, ratio_w = h / ho, w / wo
        logger.debug(
            f"init_ratio={ratio:.5f}, actual_ratio=(w={ratio_w:.5f}, h={ratio_h:.5f})"
        )
        ratio = torch.tensor([ratio_w, ratio_h])

        # 1) resize image pairs
        image = F.resize(image, size, self.interpolation)
        # 2) resize feature points
        features /= ratio

        return image, features


class ResizeValidationImageFeaturesPair(ResizeValidationImageFeatures):
    """
    Similar to `ResizeValidationImageFeatures` but resize both source and target.

    Args:
        min_size (int): The minimum allowed for the shoerter edge of the resized image.
        interpolation (InterpolationMode, optional): Desired interpolation enum defined
            by `torchvision.transforms.InterpolationMode`.
        stride (int, optional): Image must be multiply of strides, since we downsample
            the image during feature extraction.
    """

    def forward(self, item):
        source, target, affine_mat = item

        source = super().forward(source)
        target = super().forward(target)

        return source, target, affine_mat


class ToTensorValidationPair(ToTensor):
    """
    A tailored ToTensor operation for validation pairs.

    We need this custom transformationi since validation set returns
        (src_image, src_feat), (tgt_image, tgt_feat), affine_mat
    Each of them needs to convert to tensor independently.
    """

    def __call__(self, item):
        (src_image, src_feat), (tgt_image, tgt_feat), affine_mat = item

        # images, convert from (H, W, C) to (C, H, W)
        src_image = F.to_tensor(src_image)
        tgt_image = F.to_tensor(tgt_image)

        # rest of the ndarray can transform to tensor directly
        src_feat = torch.from_numpy(src_feat)
        tgt_feat = torch.from_numpy(tgt_feat)
        affine_mat = torch.from_numpy(affine_mat)

        return (src_image, src_feat), (tgt_image, tgt_feat), affine_mat
