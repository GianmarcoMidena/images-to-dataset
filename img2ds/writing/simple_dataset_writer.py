import logging
from abc import abstractmethod
from pathlib import Path

from img2ds.writing import DatasetWriter


class SimpleDatasetWriter(DatasetWriter):
    def _write_partition(self, dataset_part, output_path: Path, part_id: str = None):
        examples_counter = 0
        for id, path, label in dataset_part:
            self._write_example(id, path, label)
            examples_counter += 1
        logging.info(f"{f'partition {part_id}: ' if part_id else ''}"
                     f"{examples_counter} examples have been written.")

    @abstractmethod
    def _write_example(self, id: str, path: Path, label: str):
        ...
