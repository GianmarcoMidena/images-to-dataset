from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import tensorflow as tf
import PIL
from PIL import Image

from img2ds.writing import feature_utils as utils
from img2ds.writing.simple_tfrecords_writer import SimpleTFRecordsWriter


class ImageTFRecordsWriter(SimpleTFRecordsWriter):
    def __init__(self, n_splits: int, output_dir: Path, output_file_name: str,
                 input_archive_path: Path = None):
        super().__init__(n_splits=n_splits, output_dir=output_dir,
                         output_file_name=output_file_name)
        self._input_archive_path = input_archive_path

    def _make_example(self, id: str, path: Path, **kwargs):
        if self._input_archive_path:
            with ZipFile(self._input_archive_path) as zipfile:
                with zipfile.open(str(path)) as image_file:
                    image_bytes = image_file.read()
                    image_enc = BytesIO(image_bytes)
                    image = Image.open(image_enc)
        else:
            image = Image.open(path)
        return self._serialize_example(id, image, **kwargs)

    def _serialize_example(self, id: str, image: PIL.Image.Image, **kwargs) -> str:
        """
         Creates a tf.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        height = image.height
        width = image.width
        depth = len(image.getbands())
        image_bytes = image.tobytes()

        feature = {
            'id': utils.bytes_feature(tf.compat.as_bytes(id)),
            'image_raw': utils.bytes_feature(image_bytes),
            'height': utils.int64_feature(height),
            'width': utils.int64_feature(width),
            'depth': utils.int64_feature(depth),
        }

        if "label" in kwargs:
            feature["label"] = utils.bytes_feature(tf.compat.as_bytes(kwargs["label"]))

        for key in set(kwargs.keys()).difference({'label'}):
            value = kwargs[key]
            if isinstance(value, int) or isinstance(value, bool):
                feature[key] = utils.int64_feature(value)
            elif isinstance(value, str):
                feature[key] = utils.bytes_feature(tf.compat.as_bytes(value))
            elif isinstance(value, float):
                feature[key] = utils.float_feature(value)

        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
