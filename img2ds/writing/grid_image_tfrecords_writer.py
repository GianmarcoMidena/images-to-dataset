from pathlib import Path
from typing import List

import pandas as pd
import tensorflow as tf
from PIL import Image

from img2ds.writing import feature_utils as utils, SequenceTFRecordsWriter


class GridImageTFRecordsWriter(SequenceTFRecordsWriter):
    def _make_example(self, id: str, paths: List[Path], **kwargs):
        return self._serialize_example(id, paths, **kwargs)

    def _serialize_example(self, id: str, paths: List[Path], **kwargs) -> str:
        """
         Creates a tf.Example message ready to be written to a file.
         """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        image_filenames = pd.Series(paths).apply(str).str.rsplit(".", n=1, expand=True)[0]
        row_indices = image_filenames.str.rsplit("_", n=2, expand=True)[1].astype(int)
        col_indices = image_filenames.str.rsplit("_", n=1, expand=True)[1].astype(int)
        n_cells_per_row = col_indices.max()
        n_cells_per_col = row_indices.max()
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
            'n_cells_per_row': utils.int64_feature(n_cells_per_row),
            'n_cells_per_col': utils.int64_feature(n_cells_per_col),
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
            'row_idx': utils.int64_feature_list(row_indices),
            'col_idx': utils.int64_feature_list(col_indices),
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
