import io
import os
import zipfile
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Tuple

from torch._C import Value
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import (
    IMG_EXTENSIONS,
    default_loader,
    has_file_allowed_extension,
)
import skimage.io

default_loader = skimage.io.imread

class ZippedImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        directory: Optional[str] = "/",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[BinaryIO], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        # all subsequent path calls are confined in this zip file
        self._handle = zipfile.ZipFile(root, "r")

        super().__init__(
            directory,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        # make sure ends with '/'
        if directory[-1] != "/":
            directory += "/"
        # construct a zipfile.Path
        path = zipfile.Path(self._handle, directory)
        if not path.exists():
            raise ValueError(f"unable to locate '{directory}' in the ZIP file")

        # iterate over the directory
        for file in path.iterdir():
            print(file)
            if file.is_file():
                print(f"found file '{file}'")

                image_data = file.read_bytes()
                bytes = io.BytesIO(image_data)
                image = self.loader(bytes)
                print(image.shape)

                break

        print(self._handle.fp)

        raise RuntimeError("DEBUG")

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            # we explicitly want to use `find_classes` method
            raise ValueError("'class_to_idx' parameter cannot be None")

        # test it with an XOR
        if (extensions is None) != (is_valid_file is None):
            raise ValueError(
                "both 'extensions' and 'is_valid_file' cannot be None or not None at the same time"
            )
        if extensions is not None:
            is_valid_file = lambda x: has_file_allowed_extension(x, extensions)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index.

        Returns:
            (sample, target) where target is class_index of the target class.
        """
        pass

    def __len__(self):
        pass
