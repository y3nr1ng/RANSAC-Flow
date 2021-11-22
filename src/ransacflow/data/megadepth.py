from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Tuple

from torchvision.datasets.folder import has_file_allowed_extension

from .dataset import ZippedImageFolder


class MegaDepthTrainingDataset(ZippedImageFolder):
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
            is_valid_file = lambda x: has_file_allowed_extension(x, extensions)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            print(target_class)
            class_index = class_to_idx[target_class]
            target_dir = directory / target_class
            for file in target_dir.iterdir():
                file = file.name  # we only want str
                # if is_valid_file(file):

                item = file, class_index
                instances.append(item)

                available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            raise FileNotFoundError(
                f"found no valid file for classes {sorted(empty_classes)}"
            )

        return instances

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index)

    def __len__(self):
        return len(self.classes)


class MegaDepthValidationDataset(ZippedImageFolder):
    def __init__(self, root: Path, directory: Optional[Path] = "/", *args, **kwargs):
        # pass images directory to super
        super().__init__(root, directory / "images", *args, **kwargs)

        # TODO modify find_classes to use the matches.csv

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index)


class MegaDepthTestingDataset(ZippedImageFolder):
    def __init__(self, root: Path, directory: Optional[Path] = "/", *args, **kwargs):
        # pass images directory to super
        super().__init__(root, directory / "images", *args, **kwargs)

        # TODO modify find_classes to use the matches.csv

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index)
