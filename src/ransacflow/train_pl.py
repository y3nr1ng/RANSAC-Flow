from kornia.geometry.ransac import RANSAC
import pytorch_lightning as pl
from pytorch_lightning import Trainer

__all__ = ["RANSACFlowModel"]


class RANSACFlowModel(pl.LightningModule):
    def __init__(self):
        pass

    def configure_optimizers(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

