"""
Since we are working with our own dataset, we need to patch transforms for these custom
input formats.
"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ToTensor
from torchvision.transforms.functional_pil import affine


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
