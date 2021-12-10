import torch
import torch.nn.functional as F
import kornia


class ssim(torch.nn.Module):
    def __init__(self, window_size = 11):
        super(ssim, self).__init__()
        self.window_size = window_size
        self.reduction = 'mean'
        
    def forward(self, img1, img2, m):
        ssim_loss = kornia.losses.ssim(img1[..., m:-m, m:-m], img2[..., m:-m, m:-m], self.window_size, self.reduction)
        
        return ssim_loss


