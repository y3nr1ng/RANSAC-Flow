import io
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torchvision.datasets.folder import has_file_allowed_extension

from .dataset import ZippedImageFolder

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
            logger.warning("file integrity is not tested")

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

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # we need 2 images, randomly choose from 3
        i0, i1 = self._rng.choice(3, size=2)
        I0, I1 = super().__getitem__(i0), super().__getitem__(i1)
        return {"I0": I0, "I1": I1}  # TODO should we return dict instead of tuple?

    def __len__(self):
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
        self._matches = pd.read_csv(fp)

        # matches[scene] contains the class as integer, convert to list of str
        #   https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html
        classes = self._matches["scene"].astype("string").tolist()
        for class_idx in classes:
            class_path = directory / str(class_idx)
            if not class_path.exists():
                raise FileNotFoundError(f"could not find '{class_idx}'")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    @staticmethod
    def make_dataset(
        directory: Path,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        # TODO build dataset based on matches.csv

        pass

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index)


class MegaDepthTestingDataset(ZippedImageFolder):
    def __init__(self, root: Path, directory: Optional[Path] = "/", *args, **kwargs):
        # pass images directory to super
        super().__init__(root, directory / "images", *args, **kwargs)

        # after __init__, root now holds the opened zipfile.Path reference

        # TODO modify find_classes to use the matches.csv

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index)
