from pathlib import Path

from PIL import Image

from images_to_dataset.building import DatasetBuilder


class ImageDatasetBuilder(DatasetBuilder):
    def is_file_integral(self, file_path: Path) -> bool:
        try:
            Image.open(file_path)
            return True
        except:
            return False