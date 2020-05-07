from abc import abstractmethod
from pathlib import Path

import tensorflow as tf

from images_to_dataset.writing import DatasetWriter


class TFRecordsWriter(DatasetWriter):
    def _write_partition(self, dataset_part, output_path: Path, part_id: str = None):
        with tf.io.TFRecordWriter(str(output_path)) as self._writer:
            super()._write_partition(dataset_part=dataset_part, output_path=output_path, part_id=part_id)

    def _write_example(self, path, label):
        example = self._make_example(path, label)
        self._writer.write(example)

    def _get_extension(self) -> str:
        return "tfrecords"

    @abstractmethod
    def _make_example(self, path, label):
        ...
