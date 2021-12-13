import itertools

import torch

import pytorch_lightning as pl
from kornia.geometry.ransac import RANSAC
from pytorch_lightning import Trainer

from .model import CorrNeigh, FeatureExtractor, NetFlow

__all__ = [
    "RANSACFlowModelStage1",
    "RANSACFlowModelStage2",
    "RANSACFlowModelStage3",
    "RANSACFlowModelStage4",
]


class RANSACFlowModel(pl.LightningModule):
    def __init__(self, kernel_size: int, lr: float):
        super().__init__()

        self.lr = lr
        self.kernel_size = kernel_size

        # FIXME consolidate NetFlow
        # FIXME CorrNeight probably does not have learnable parameter, look this up
        self.feature_extractor = FeatureExtractor()
        self.correlator = CorrNeigh(kernel_size)
        self.coarse_flow = NetFlow(kernel_size, "netFlowCoarse")
        self.fine_flow = NetFlow(kernel_size, "netMatch")

        # TODO how to load weights (in later stages?)

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass


class RANSACFlowModelStage1(RANSACFlowModel):
    def configure_optimizers(self):
        coarse_flow_params = itertools.chain(
            self.feature_extractor.parameters(),
            self.correlator.parameters(),
            self.coarse_flow.parameters(),
        )
        coarse_flow_opt = torch.optim.Adam(
            coarse_flow_params, lr=self.lr, betas=(0.5, 0.999)
        )
        return coarse_flow_opt

    def training_step(self, batch, batch_idx):
        pass


class RANSACFlowModelStage2(RANSACFlowModel):
    def configure_optimizers(self):
        pass

    def training_step(self, batch, batch_idx):
        pass


class RANSACFlowModelStage3(RANSACFlowModel):
    def configure_optimizers(self):
        """Jointly train the coarse and fine flow network."""
        coarse_flow_params = itertools.chain(
            self.feature_extractor.parameters(),
            self.correlator.parameters(),
            self.coarse_flow.parameters(),
        )
        coarse_flow_opt = torch.optim.Adam(
            coarse_flow_params, lr=self.lr, betas=(0.5, 0.999)
        )

        fine_flow_opt = torch.optim.Adam(
            self.fine_flow.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )

        return (coarse_flow_opt, fine_flow_opt)

    def training_step(self, batch, batch_idx):
        pass


class RANSACFlowModelStage4(RANSACFlowModel):
    def configure_optimizers(self):
        """For better visual results, smooth the flow to reduce distortions. """
        coarse_flow_opt = torch.optim.Adam(
            self.coarse_flow.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )
        return coarse_flow_opt

    def training_step(self, batch, batch_idx):
        pass

