from kornia.geometry.ransac import RANSAC
import pytorch_lightning as pl
from pytorch_lightning import Trainer

__all__ = ["RANSACFlowModel"]


class RANSACFlowModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # TODO create all 4 models

        # TODO how to load weights (in later stages?)

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass


class RANSACFlowModelStage1(RANSACFlowModel):
    def configure_optimizers(self):
        pass


class RANSACFlowModelStage2(RANSACFlowModel):
    def configure_optimizers(self):
        pass


class RANSACFlowModelStage3(RANSACFlowModel):
    def configure_optimizers(self):
        pass


class RANSACFlowModelStage4(RANSACFlowModel):
    pass
