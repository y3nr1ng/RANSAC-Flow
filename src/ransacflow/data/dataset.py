import io
import mmap
from collections import defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Tuple
from zipfile import ZipFile

import skimage.io
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import has_file_allowed_extension

__all__ = ["ZippedImageFolder"]

default_loader = skimage.io.imread


class wrapped_mmap(mmap.mmap):
    """
    For some magical reason, `zipfile.ZipFile` wraps incoming file-like object in
    `_SharedFile`, which further requires the object to be `seekable`, as in `IOBase`
    definition. However, `mmap` object does not have this function, this wrapper is a
    hacky attempt to patch this.

    Reference:
        https://github.com/python/cpython/blob/7c5b01b5101923fc38274c491bd55239ee9f0416/
        Lib/zipfile.py#L744
    """

    def seekable(self) -> bool:
        """Return True if the stream supports random access."""
        return True


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
        # using memmory mapped file handle to ensure this works with multiprocess
        fd = open(root, mode="rb")
        fd_mapped = wrapped_mmap(fd.fileno(), length=0, access=mmap.ACCESS_READ)
        self._handle = ZipFile(fd_mapped, "r")

        # NOTE necessary evil to leak the ZIP handle as private member, currently don't
        # have better way to expose this in __getitem__
        zip_path = (self._handle, Path(directory))

        super().__init__(
            zip_path,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )

    def find_classes(
        self, zip_path: Tuple[ZipFile, Path]
    ) -> Tuple[List[str], Dict[str, int]]:
        """
        Find class folders in a dataset.

        This method follows the original implementation, but operates inside a ZIP file.

        Args:
            zip_path (tuple of (ZipFile, Path)):
                A opened zip file and root path in the file.

        Raises:
            FileNotFounderror: If `directory` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes, and a dictionary
                mapping each class to an index.
        """
        handle, directory = zip_path

        classes = set()
        for file in handle.namelist():
            try:
                path = Path(file).relative_to(directory)
                classes.add(path.parent)
            except ValueError:
                # does not belong to `directory`
                pass

        classes = list(classes)
        if not classes:
            raise FileNotFoundError(f"could not find any class folder in '{directory}'")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(
        zip_path: Tuple[ZipFile, Path],
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """
        Generate a list of samples of a form (path_to_sample, class).

        This method follows the original implementation, but operates inside a ZIP file.

        Args:
            zip_path (tuple of (ZipFile, Path)):
                A opened zip file and root path in the file.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (Tuple[int, ...], optional): A list of allowed extensions.
            is_valid_file (Callable[[str], bool], optional): A function that takes path
                of a file and checks if it is a valid file.
        """
        handle, directory = zip_path

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

        files = list(handle.infolist())

        instances = defaultdict(list)
        for file in files:
            # NOTE scan over members again, but this time we will only process further
            # for those in class_to_idx list, allowing some customization to remove
            # unwanted classes
            try:
                path = Path(file.filename).relative_to(directory)
            except ValueError:
                continue
            else:
                target_class = path.parent
                if target_class not in class_to_idx:
                    continue

            if is_valid_file(file.filename):
                instances[target_class].append(file)

        empty_classes = set(class_to_idx.keys()) - set(instances.keys())
        if empty_classes:
            raise FileNotFoundError(
                f"found no valid file for classes {sorted(empty_classes)}"
            )

        # convert dictionary of lists to list of key-value pairs
        instances = [(v, k) for k in instances for v in instances[k]]
        return instances

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index.

        Returns:
            (sample, target) where target is class_index of the target class.
        """
        file, target = self.samples[index]

        stream = self._handle.open(file, "r")
        sample = self.loader(stream)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
