from pathlib import Path

from PIL import Image

from img2ds.building import DatasetBuilder


class ImageDatasetBuilder(DatasetBuilder):
    def check_data_integrity(self, file_path: Path) -> bool:
        try:
            Image.open(file_path)
            return True
        except:
            return False