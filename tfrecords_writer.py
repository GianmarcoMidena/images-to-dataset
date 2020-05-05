import logging
from abc import abstractmethod, ABC
from pathlib import Path

import tensorflow as tf


class TFRecordsWriter(ABC):
    def __init__(self, n_splits: int, output_dir: Path):
        self._n_splits = n_splits
        self._output_dir = output_dir

    def write(self, partitions):
        self._output_dir.mkdir(exist_ok=True)
        if self._n_splits > 1:
            for i, part in enumerate(partitions):
                self._write_partition(dataset_part=part,
                                      output_path=self._output_dir.joinpath(f"data_part_{i + 1}.tfrecords"))
        else:
            for dataset in partitions:
                self._write_partition(dataset_part=dataset,
                                      output_path=self._output_dir.joinpath("data.tfrecords"))

    def _write_partition(self, dataset_part, output_path: Path):
        with tf.io.TFRecordWriter(str(output_path)) as writer:
            examples_counter = 0
            for path, label in dataset_part:
                example = self._make_example(path, label)
                writer.write(example)
                examples_counter += 1
            logging.info(f"{examples_counter} examples have been written.")

    @abstractmethod
    def _make_example(self, path, label):
        ...
