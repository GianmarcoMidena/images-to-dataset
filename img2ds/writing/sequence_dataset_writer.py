import logging
from abc import abstractmethod
from pathlib import Path

from typing import List

from img2ds.writing import DatasetWriter


class SequenceDatasetWriter(DatasetWriter):
    def _write_partition(self, dataset_part, output_path: Path, part_id: str = None):
        examples_counter = 0
        for id, paths, label, kwargs in dataset_part:
            self._write_example(id, paths, label, **kwargs)
            examples_counter += 1
        logging.info(f"{f'partition {part_id}: ' if part_id else ''}"
                     f"{examples_counter} examples have been written.")

    @abstractmethod
    def _write_example(self, id: str, paths: List[Path], label: str, **kwargs):
        ...
