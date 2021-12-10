import io
import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.random.mtrand import shuffle
import pandas as pd
import pytorch_lightning as pl
import torch
from torchvision.datasets.folder import has_file_allowed_extension

from .dataset import ZippedImageFolder

__all__ = ["MegaDepthDataModule"]

logger = logging.getLogger("ransacflow.data.megadepth")


class MegaDepthTrainingDataset(ZippedImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng()

    @staticmethod
    def make_dataset(
        directory: Path,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        if class_to_idx is None:
            # we explicitly want to use `find_classes` method
            raise ValueError("'class_to_idx' parameter cannot be None")

        if not ((extensions is None) ^ (is_valid_file is None)):
            raise ValueError(
                "both 'extensions' and 'is_valid_file' cannot be None or not None at the same time"
            )
        if extensions is not None:
            # x should be a Path-like object
            logger.warning("file integrity is not explicitly tested")

        instances = []
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = directory / target_class

            # we know each training set has 3 images, hard coded it to load them
            for i in range(3):
                file = target_dir / f"{i+1}.jpg"
                item = file, class_index
                instances.append(item)

        return instances

    def __getitem__(self, index: int):
        # we need 2 images, randomly choose from [i+0]-[i+2] (3 images)
        offsets = self._rng.choice(3, size=2)
        image_pair = tuple(super().__getitem__(index + offset) for offset in offsets)
        return image_pair

    def __len__(self) -> int:
        return len(self.classes)


class MegaDepthValidationDataset(ZippedImageFolder):
    def __init__(self, root: Path, directory: Optional[Path] = "/", *args, **kwargs):
        # pass images directory to super
        directory = Path(directory)
        super().__init__(root, directory / "images", *args, **kwargs)

    def find_classes(self, directory: Path) -> Tuple[List[str], Dict[str, int]]:
        """
        Find class folders in a dataset.

        This method follows the original implementation, but operates inside a ZIP file.

        Args:
            directory (Path): Directory path inside the ZIP file.

        Raises:
            FileNotFounderror: If `directory` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes, and a dictionary
                mapping each class to an index.
        """
        # normalize directory with trailing slash
        #   https://bugs.python.org/issue21039
        directory /= ""
        if not directory.exists():
            raise ValueError(f"unable to locate '{directory}' in the ZIP file")

        # after VisionDataset.__init__, root holds the zipfile.Path to images
        # but, our match list is one folder upward
        path = directory.parent / "matches.csv"
        fp = io.BytesIO(path.read_bytes())
        matches = pd.read_csv(fp)

        # matches[scene] contains the class as integer, convert to list of str
        #   https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html
        classes = matches["scene"].astype("string").tolist()
        classes = list(set(classes))
        for cls_name in classes:
            class_path = directory / cls_name
            if not class_path.exists():
                raise FileNotFoundError(f"could not find '{cls_name}'")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    @staticmethod
    def make_dataset(
        directory: Path,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        if class_to_idx is None:
            # we explicitly want to use `find_classes` method
            raise ValueError("'class_to_idx' parameter cannot be None")

        if not ((extensions is None) ^ (is_valid_file is None)):
            raise ValueError(
                "both 'extensions' and 'is_valid_file' cannot be None or not None at the same time"
            )
        if extensions is not None:
            # x should be a Path-like object
            logger.warning("file integrity is not explicitly tested")

        # normalize directory with trailing slash
        #   https://bugs.python.org/issue21039
        directory /= ""

        # we don't want to work with DataFrame, convert them all to ndarray
        path = directory.parent / "matches.csv"
        fp = io.BytesIO(path.read_bytes())
        matches = pd.read_csv(fp)
        # convert some units first
        matches["scene"] = matches["scene"].astype("string")

        # load ground truth transformation matrix
        path = directory.parent / "affine.pkl"
        with path.open("r") as fp:
            # it was stored as a dictionary
            affine_mats = pickle.load(fp)

        instances = []
        for (_, row), affine_mat in zip(matches.iterrows(), affine_mats.values()):
            # class name
            cls_name = row["scene"]
            class_index = class_to_idx[cls_name]

            # zipfile.Path to image
            src_path, tgt_path = row["source_image"], row["target_image"]
            src_path = directory / cls_name / src_path
            tgt_path = directory / cls_name / tgt_path

            # load and compact feature coordinates
            src_feat_x = np.fromstring(row["XA"], dtype=np.float32, sep=";")
            src_feat_y = np.fromstring(row["YA"], dtype=np.float32, sep=";")
            src_feat = np.stack([src_feat_y, src_feat_x], axis=-1)
            tgt_feat_x = np.fromstring(row["XB"], dtype=np.float32, sep=";")
            tgt_feat_y = np.fromstring(row["YB"], dtype=np.float32, sep=";")
            tgt_feat = np.stack([tgt_feat_y, tgt_feat_x], axis=-1)

            # NOTE
            # ground truth affine transformation matrix is stored directly

            item = (src_path, src_feat), (tgt_path, tgt_feat), affine_mat, class_index
            instances.append(item)

        return instances

    def __getitem__(self, index: int):
        source, target, affine_mat, _ = self.samples[index]
        src_path, src_feat = source
        tgt_path, tgt_feat = target

        # load source and target images
        fp = io.BytesIO(src_path.read_bytes())
        src_image = self.loader(fp)
        fp = io.BytesIO(tgt_path.read_bytes())
        tgt_image = self.loader(fp)

        if self.transform is not None:
            src_image, src_feat = self.transform(src_image, src_feat)
        if self.target_transform is not None:
            tgt_image, tgt_feat = self.target_transform(tgt_image, tgt_feat)

        return (src_path, src_feat), (tgt_path, tgt_feat), affine_mat


class MegaDepthDataModule(pl.LightningDataModule):
    """[summary]

    Args:
        path (Path): Path to the ZIP file.
        batch_size (int, optional): How many samples per batch to load.
    """

    def __init__(self, path: Path, batch_size: int = 16):
        super().__init__()

        self.path = path
        self.batch_size = batch_size

    def setup(self):
        self.megadepth_train = MegaDepthTrainingDataset(self.path, "train")
        self.megadepth_val = MegaDepthValidationDataset(self.path, "validate")

    def train_dataloader(self):
        """
        root: Path,
        directory: Optional[Path] = "/",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[BinaryIO], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        """
        megadepth_train = torch.utils.data.DataLoader(
            self.megadepth_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        return megadepth_train

    def val_dataloader(self):
        # TODO should we drop_last for validation?
        megadepth_val = torch.utils.data.DataLoader(
            self.megadepth_val,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )
        return megadepth_val
