import itertools

import pytorch_lightning as pl
import torch
from kornia.geometry.ransac import RANSAC
from pytorch_lightning import Trainer

from .model import (
    FeatureExtractor,
    NeighborCorrelator,
    FlowPredictor,
    MatchabilityPredictor,
)

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
        gamma (float): Weight for the gradient. FIXME what the fuck is this? stage4?
        image_size (int): FIXME TBD, we assume it is square
        kernel_size (int): FIXME TBD, we assume it is square
        lr (float): Learning rate.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        image_size: int,
        kernel_size: int,
        lr: float,
        pretrained: bool = True,
    ):
        super().__init__()

        # FIXME NeighborCorrelator does not have learnable parameter, look this up
        self.feature_extractor = FeatureExtractor(pretrained=pretrained)
        self.correlator = NeighborCorrelator(kernel_size)
        self.flow = FlowPredictor(image_size, kernel_size)
        self.matchability = MatchabilityPredictor(image_size, kernel_size)

        # FIXME how to load weights (in later stages?)

        # FIXME set non-trainalbe network to eval()

        # save everything passes to __init__ as hyperparameters, self.hparams
        # https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # extract feature correlation map
        print(len(batch))

        raise RuntimeError("DEBUG")

        # coarse flow prediction

        # fine flow prediction

        # calculate losses

    def validation_step(self, batch, batch_idx):
        pass


class RANSACFlowModelStage1(RANSACFlowModel):
    def configure_optimizers(self):
        coarse_flow_params = itertools.chain(
            self.feature_extractor.parameters(), self.flow.parameters(),
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
