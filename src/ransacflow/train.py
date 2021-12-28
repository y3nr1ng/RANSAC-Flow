import itertools
import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .model import (
    FeatureExtractor,
    FlowPredictor,
    MatchabilityPredictor,
    NeighborCorrelator,
)
from .model.loss import ReconstructionLoss

__all__ = [
    "RANSACFlowModelStage1",
    "RANSACFlowModelStage2",
    "RANSACFlowModelStage3",
    "RANSACFlowModelStage4",
]

logger = logging.getLogger("ransacflow.train")


class RANSACFlowModel(pl.LightningModule):
    """
    All stages are simply training with different losses. Therefore, each stage subclasses this module, and implement their version of concrete `training_step`.

    Args:
        alpha (float): Weight for matchability loss.
        beta (float): Weight for cycle consistency loss.
        gamma (float): Weight for the gradient. FIXME what the fuck is this? stage4?
        kernel_size (int): FIXME TBD, we assume it is square
        lr (float): Learning rate.
        pretrained (bool, optional): Use pretrained model.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        kernel_size: int,
        ssim_window_size: int,
        lr: float,
        pretrained: bool = True,
    ):
        super().__init__()

        assert kernel_size % 2 == 1, "kernel size has to be odd"

        assert ssim_window_size % 2 == 1, "SSIM window size has to be odd"

        # instantiate our networks
        self.feature_extractor = FeatureExtractor(pretrained=pretrained)
        self.correlator = NeighborCorrelator(kernel_size)
        self.flow = FlowPredictor(kernel_size)
        self.matchability = MatchabilityPredictor(kernel_size)

        # FIXME how to load weights (in later stages?)

        # FIXME set non-trainalbe network to eval()

        # scaling parameters and grids used during flow generation
        self.image_size = (-1, -1)
        self.register_buffer("scale", torch.tensor([]), persistent=False)
        self.register_buffer("grid", torch.tensor([]), persistent=False)

        # histogram for validation error
        error_ticks = np.logspace(0, np.log10(36), num=8).round()
        error_ticks = torch.tensor(error_ticks)
        self.register_buffer("error_ticks", error_ticks, persistent=False)
        self.register_buffer(
            "error_counts", torch.zeros_like(error_ticks), persistent=False
        )

        # save everything passes to __init__ as hyperparameters, self.hparams
        # https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
        self.save_hyperparameters()

    def forward(self, I_s, I_t):
        """
        Given image pairs, generate flows that match samples to targets.

        Args:
            I_s (TBD): TBD
            I_t (TBD): TBD
        """
        assert (
            I_s.shape == I_t.shape
        ), f"images have different shapes, {I_s.shape} and {I_t.shape}"
        if self.image_size != I_s.shape[-2:]:
            self.image_size = I_s.shape[-2:]

            # scale flow vectors
            scale = torch.tensor(self.image_size[::-1], device=self.scale.device)
            scale = scale / 2.0
            self.scale = scale.view(1, -1, 1, 1)

            # recreate grid
            logger.debug(f"create new grid {self.image_size}")
            ny, nx = self.image_size
            vx = torch.linspace(-1, 1, nx, device=self.grid.device)
            vy = torch.linspace(-1, 1, ny, device=self.grid.device)
            grid_x, grid_y = torch.meshgrid(vx, vy, indexing="xy")
            # NOTE F.grid_sample() expects grid dimension (B, H, W, 2)
            self.grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        # extract feature correlation map
        f_s = self.feature_extractor(I_s)
        f_t = self.feature_extractor(I_t)

        # calculate cosine similarity
        s_st = self.correlator(f_s, f_t)

        # estimate flow
        F_st = self.flow(s_st)
        F_st = F.interpolate(F_st, size=tuple(self.image_size), mode="bilinear")
        F_st /= self.scale

        # since flow is generally use as grid sampler, we permute its axes to follow the
        # convention uesd in F.grid_sample()
        F_st = F_st.permute(0, 2, 3, 1)

        # convert flow from local offset to global coordinate, and clip to [-1, 1]
        F_st = torch.clamp(F_st + self.grid, min=-1, max=1)

        return F_st

    def validation_step(self, batch, batch_idx):
        # NOTE we don't really have a batch here, refer to commit 555371
        (I_s, src_feat), (I_t, tgt_feat), affine_mat = batch

        # align image using affine matrix
        F_affine = F.affine_grid(affine_mat, I_s.shape)
        I_s_warped = F.grid_sample(I_s, F_affine)

        # predict flow between I_t and F_0(I_s) ~ I_s_warped
        # I_t and I_s_warped should closely match each other
        F_ts = self(I_t, I_s_warped)

        # correct the flow from affine transformation
        F_corrected = F.grid_sample(F_affine.permute(0, 3, 1, 2), F_ts)
        F_corrected = F_corrected.permute(0, 2, 3, 1)

        # estimated flow is [-1, 1], convert it to (target pixel) [0, n) coordinate
        scale = torch.tensor(I_s.shape[-2:][::-1], device=F_corrected.device) / 2.0
        F_corrected = (F_corrected + 1) * scale

        # extract estimated source feature points
        tgt_feat = torch.round(tgt_feat).long()
        tgt_feat_x = tgt_feat[..., 0]
        tgt_feat_y = tgt_feat[..., 1]
        src_feat_F = F_corrected[:, tgt_feat_y, tgt_feat_x, :]
        # H and W dimension becomes (alternative) N dimension, the actual N dimension
        # is fixed to 1, so we need to squeeze dim 1 to match source tensor
        src_feat_F = src_feat_F.squeeze(1)

        # calculate alignment error
        diff = src_feat - src_feat_F
        diff = torch.hypot(diff[..., 0], diff[..., 1])

        # accumulate errors to histogram
        counts = diff.view(-1, 1) < self.error_ticks
        counts = torch.sum(counts, dim=0, keepdim=False)
        print(f"counts.shape={counts.shape}")
        print(f"counts={counts}")

        raise RuntimeError("DEBUG, base, validation_step, non-iterative")

        ## iterative
        # NOTE though batch is 1, we still keep it here for future work
        src_feat = src_feat.swapaxes(0, 1)
        tgt_feat = tgt_feat.swapaxes(0, 1)
        px_diff = []
        for src_pt, tgt_pt in zip(src_feat, tgt_feat):
            print(f"src={src_pt}, tgt={tgt_pt}")

            # TODO index into F_corrected, and compare pixel shift differences

            # NOTE in the original paper, they round feature points so they can index
            # into the flow F without interpolation
            src_pt = torch.round(src_pt)
            tgt_pt = torch.round(tgt_pt)

            # TODO is this correct?
            tgt_x, tgt_y = tgt_pt.long().T
            src_pt_F = F_corrected[:, tgt_y, tgt_x, :]

            print(f"src={src_pt}, tgt={tgt_pt}, F(src)={src_pt_F}")

            raise RuntimeError("DEBUG, base, validation_step")


class RANSACFlowModelStage1(RANSACFlowModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.loss_rec = ReconstructionLoss(self.hparams.ssim_window_size)

    def configure_optimizers(self):
        coarse_flow_params = itertools.chain(
            self.feature_extractor.parameters(), self.flow.parameters(),
        )
        coarse_flow_opt = torch.optim.Adam(
            coarse_flow_params, lr=self.hparams.lr, betas=(0.5, 0.999)
        )
        return coarse_flow_opt

    def training_step(self, batch, batch_idx):
        # we are giving an image pair, a sample image I_s, and a target image I_t
        # swapping them will gives a "new" pair of matching set, so we stack them as if
        # we get doubled batch size
        I_st = torch.cat(batch, dim=0)
        # due to aforementioned reason, we need a set of index to flip the batch axes
        batch_size = batch[0].shape[0]
        batch_flipped = torch.roll(torch.arange(2 * batch_size), batch_size)

        # NOTE
        #   all annotations from now on are based on first half of the batch
        I_ts = I_st[batch_flipped]

        # predict flow
        F_st = self(I_st, I_ts)

        # I_st_warped = F_st(I_st) ~ I_ts
        I_st_warped = F.grid_sample(I_st, F_st)

        loss = self.loss_rec(I_st_warped, I_ts)
        self.log("loss_rec", loss)

        return loss


class RANSACFlowModelStage2(RANSACFlowModel):
    def configure_optimizers(self):
        pass

    def training_step(self, batch, batch_idx):
        # we are giving an image pair, a sample image I_s, and a target image I_t
        # swapping them will gives a "new" pair of matching set, so we stack them as if
        # we get doubled batch size
        I_st = torch.cat(batch, dim=0)
        # due to aforementioned reason, we need a set of index to flip the batch axes
        batch_size = batch[0].shape[0]
        batch_flipped = torch.roll(torch.arange(2 * batch_size), batch_size)

        # NOTE
        #   all annotations from now on are based on first half of the batch
        I_ts = I_st[batch_flipped]

        # predict flow
        F_st = self(I_st, I_ts)
        F_ts = F.grid_sample(F_st[batch_flipped], F_st)
        print(F.shape)


class RANSACFlowModelStage3(RANSACFlowModel):
    def configure_optimizers(self):
        """Jointly train the coarse and fine flow network."""
        # coarse flow and its optimizer
        coarse_flow_params = itertools.chain(
            self.feature_extractor.parameters(), self.flow.parameters(),
        )
        coarse_flow_opt = torch.optim.Adam(
            coarse_flow_params, lr=self.hparams.lr, betas=(0.5, 0.999)
        )

        # fine flow and its optimizer (we only have 1 network)
        fine_flow_opt = torch.optim.Adam(
            self.matchability.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999)
        )

        return (coarse_flow_opt, fine_flow_opt)

    def training_step(self, batch, batch_idx):
        pass

        # computeLossMatchability


class RANSACFlowModelStage4(RANSACFlowModel):
    def configure_optimizers(self):
        """For better visual results, smooth the flow to reduce distortions. """
        coarse_flow_opt = torch.optim.Adam(
            self.flow.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999)
        )
        return coarse_flow_opt

    def training_step(self, batch, batch_idx):
        pass

        # computeLossMatchability
