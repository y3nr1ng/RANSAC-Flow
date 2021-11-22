import io
import os
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Tuple

import skimage.io
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader, has_file_allowed_extension

default_loader = skimage.io.imread


class ZippedImageFolder(ImageFolder):
    def __init__(
        self,
        root: Path,
        directory: Optional[Path] = "/",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[BinaryIO], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        # all subsequent path calls are confined in this zip file
        self._handle = zipfile.ZipFile(root, "r")
        directory = zipfile.Path(self._handle, directory)

        # while we specify `Path` as directory type, we are actually working with #
        # zipfile.Path for zipped folder
        super().__init__(
            directory,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )

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
        directory /= ''
        if not directory.exists():
            raise ValueError(f"unable to locate '{directory}' in the ZIP file")

        classes = sorted(entry.name for entry in directory.iterdir() if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"could not find any class folder in '{directory}'")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    @staticmethod
    def make_dataset(
        directory: Path,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """
        Generate a list of samples of a form (path_to_sample, class).

        This method follows the original implementation, but operates inside a ZIP file.

        Args:
            directory (Path): Directory path inside the ZIP file.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (Tuple[int, ...], optional): A list of allowed extensions.
            is_valid_file (Callable[[str], bool], optional): A function that takes path
                of a file and checks if it is a valid file.
        """
        if class_to_idx is None:
            # we explicitly want to use `find_classes` method
            raise ValueError("'class_to_idx' parameter cannot be None")

        if not ((extensions is None) ^ (is_valid_file is None)):
            raise ValueError(
                "both 'extensions' and 'is_valid_file' cannot be None or not None at the same time"
            )
        if extensions is not None:
            # x should be a Path-like object
            is_valid_file = lambda x: has_file_allowed_extension(x.name, extensions)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = directory / target_class
            if not target_dir.is_dir():
                continue
            for file in target_dir.iterdir():
                if is_valid_file(file):
                    item = file, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            raise FileNotFoundError(
                f"found no valid file for classes {sorted(empty_classes)}"
            )

        return instances

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index.

        Returns:
            (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
