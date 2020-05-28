from pathlib import Path
from typing import List

import PIL
import tensorflow as tf
from PIL import Image

from img2ds.writing import feature_utils as utils, SequenceTFRecordsWriter


class SequenceImageTFRecordsWriter(SequenceTFRecordsWriter):
    def _make_example(self, id: str, paths: List[Path], label: str):
        images = [Image.open(path) for path in paths]
        return self._serialize_example(id, images, label)

    def _serialize_example(self, id: str, images: List[PIL.Image.Image], label: str) -> str:
        """
         Creates a tf.Example message ready to be written to a file.
         """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        heights = [image.height for image in images]
        widths = [image.width for image in images]
        depths = [len(image.getbands()) for image in images]
        image_bytes = [image.tobytes() for image in images]

        context = {
            'id': utils.bytes_feature(tf.compat.as_bytes(id)),
            'label': utils.bytes_feature(tf.compat.as_bytes(label)),
        }

        feature_list = {
            'image_raw': utils.bytes_feature_list(image_bytes),
            'height': utils.int64_feature_list(heights),
            'width': utils.int64_feature_list(widths),
            'depth': utils.int64_feature_list(depths),
        }

        # Create a Features message using tf.train.SequenceExample.
        example_proto = tf.train.SequenceExample(context=tf.train.Features(feature=context),
                                                 feature_lists=tf.train.FeatureLists(feature_list=feature_list))
        return example_proto.SerializeToString()
