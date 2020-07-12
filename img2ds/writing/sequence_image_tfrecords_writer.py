from pathlib import Path
from typing import List, Dict, Union

import tensorflow as tf
from PIL import Image

from img2ds.writing import feature_utils as utils, SequenceTFRecordsWriter


class SequenceImageTFRecordsWriter(SequenceTFRecordsWriter):
    def _make_example(self, id: str, paths: List[Path], **kwargs):
        return self._serialize_example(id, paths, **kwargs)

    def _serialize_example(self, id: str, paths: List[Path], **kwargs) -> str:
        """
         Creates a tf.Example message ready to be written to a file.
         """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        heights = []
        widths = []
        depths = []
        image_bytes = []
        for path in paths:
            with Image.open(path) as image:
                heights.append(image.height)
                widths.append(image.width)
                depths.append(len(image.getbands()))
                image_bytes.append(image.tobytes())

        context = {
            'id': utils.bytes_feature(tf.compat.as_bytes(id)),
        }

        if "global_label" in kwargs:
            context["label"] = utils.bytes_feature(tf.compat.as_bytes(kwargs["global_label"]))
        elif "label" in kwargs:
            context["label"] = utils.bytes_feature(tf.compat.as_bytes(kwargs["label"]))

        for key in set(kwargs.keys()).difference({'label', 'global_label', 'local_label'}):
            value = kwargs[key]
            if isinstance(value, int) or isinstance(value, bool):
                context[key] = utils.int64_feature(value)
            elif isinstance(value, str):
                context[key] = utils.bytes_feature(tf.compat.as_bytes(value))
            elif isinstance(value, float):
                context[key] = utils.float_feature(value)

        feature_list = {
            'height': utils.int64_feature_list(heights),
            'width': utils.int64_feature_list(widths),
            'depth': utils.int64_feature_list(depths),
            'image_raw': utils.bytes_feature_list(image_bytes),
        }

        if "local_label" in kwargs:
            feature_list["label"] = utils.bytes_feature_list([tf.compat.as_bytes(l) for l in kwargs["local_label"]])

        # Create a Features message using tf.train.SequenceExample.
        example_proto = tf.train.SequenceExample(context=tf.train.Features(feature=context),
                                                 feature_lists=tf.train.FeatureLists(feature_list=feature_list))
        return example_proto.SerializeToString()
