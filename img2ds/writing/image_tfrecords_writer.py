from pathlib import Path

import tensorflow as tf
import PIL
from PIL import Image

from img2ds.writing import feature_utils as utils
from img2ds.writing.simple_tfrecords_writer import SimpleTFRecordsWriter


class ImageTFRecordsWriter(SimpleTFRecordsWriter):
    def _make_example(self, id: str, path: Path, label: str):
        image = Image.open(path)
        return self._serialize_example(id, image, label)

    def _serialize_example(self, id: str, image: PIL.Image.Image, label: str) -> str:
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
            'label': utils.bytes_feature(tf.compat.as_bytes(label)),
            'image_raw': utils.bytes_feature(image_bytes),
            'height': utils.int64_feature(height),
            'width': utils.int64_feature(width),
            'depth': utils.int64_feature(depth),
        }

        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
