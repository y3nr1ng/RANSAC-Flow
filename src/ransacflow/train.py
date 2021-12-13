import itertools

import pytorch_lightning as pl
import torch
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
    """
    Args:
        alpha (float): Weight for matchability loss.
        beta (float): Weight for cycle consistency loss.
        kernel_size (int): TBD
        lr (float): Learning rate.
    """

    def __init__(self, alpha: float, beta: float, kernel_size: int, lr: float):
        super().__init__()

        # FIXME consolidate NetFlow
        # FIXME CorrNeight probably does not have learnable parameter, look this up
        self.feature_extractor = FeatureExtractor()
        self.correlator = CorrNeigh(kernel_size)
        self.coarse_flow = NetFlow(kernel_size, "netFlowCoarse")
        self.fine_flow = NetFlow(kernel_size, "netMatch")

        # TODO how to load weights (in later stages?)

        # save everything passes to __init__ as hyperparameters, self.hparams
        # https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
        self.save_hyperparameters()

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
            coarse_flow_params, lr=self.hparams.lr, betas=(0.5, 0.999)
        )
        return coarse_flow_opt

    def training_step(self, batch, batch_idx):
        pass

        # computeLossNoMatchability


class RANSACFlowModelStage2(RANSACFlowModel):
    def configure_optimizers(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

        # computeLossNoMatchability


class RANSACFlowModelStage3(RANSACFlowModel):
    def configure_optimizers(self):
        """Jointly train the coarse and fine flow network."""
        coarse_flow_params = itertools.chain(
            self.feature_extractor.parameters(),
            self.correlator.parameters(),
            self.coarse_flow.parameters(),
        )
        coarse_flow_opt = torch.optim.Adam(
            coarse_flow_params, lr=self.hparams.lr, betas=(0.5, 0.999)
        )

        fine_flow_opt = torch.optim.Adam(
            self.fine_flow.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )

        return (coarse_flow_opt, fine_flow_opt)

    def training_step(self, batch, batch_idx):
        pass

        # computeLossMatchability


class RANSACFlowModelStage4(RANSACFlowModel):
    def configure_optimizers(self):
        """For better visual results, smooth the flow to reduce distortions. """
        coarse_flow_opt = torch.optim.Adam(
            self.coarse_flow.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999)
        )
        return coarse_flow_opt

    def training_step(self, batch, batch_idx):
        pass

        # computeLossMatchability
