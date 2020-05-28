from pathlib import Path

import tensorflow as tf

from img2ds.writing import DatasetWriter


class TFRecordsWriter(DatasetWriter):
    def _write_partition(self, dataset_part, output_path: Path, part_id: str = None):
        with tf.io.TFRecordWriter(str(output_path)) as self._writer:
            super()._write_partition(dataset_part=dataset_part, output_path=output_path, part_id=part_id)

    def _get_extension(self) -> str:
        return "tfrecords"
