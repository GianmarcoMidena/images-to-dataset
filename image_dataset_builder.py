from pathlib import Path

from PIL import Image

from dataset_builder import DatasetBuilder


class ImageDatasetBuilder(DatasetBuilder):
    def __init__(self, data_root: Path, n_splits: int, shuffle: bool, stratify: bool, seed: int = None):
        super().__init__(data_root=data_root, n_splits=n_splits, shuffle=shuffle, stratify=stratify, seed=seed)

    def is_file_integral(self, file_path: Path) -> bool:
        try:
            Image.open(file_path)
            return True
        except:
            return False