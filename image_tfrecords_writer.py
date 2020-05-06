from pathlib import Path

import tensorflow as tf
import PIL
from PIL import Image

from tfrecords_writer import TFRecordsWriter


class ImageTFRecordsWriter(TFRecordsWriter):
    def _make_example(self, path, label):
        image = Image.open(path)
        return self._serialize_example(image, label)

    def _serialize_example(self, image: PIL.Image.Image, label: str) -> str:
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
            'height': self._int64_feature(height),
            'width': self._int64_feature(width),
            'depth': self._int64_feature(depth),
            'label': self._bytes_feature(tf.compat.as_bytes(label)),
            'image_raw': self._bytes_feature(image_bytes)
        }

        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
