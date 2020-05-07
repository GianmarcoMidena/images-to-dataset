import logging
from abc import abstractmethod, ABC
from pathlib import Path


class DatasetWriter(ABC):
    def __init__(self, n_splits: int, output_dir: Path, output_file_name: str):
        self._n_splits = n_splits
        self._output_dir = output_dir
        self._output_file_name = output_file_name

    def write(self, partitions):
        self._output_dir.mkdir(exist_ok=True)
        if self._n_splits > 1:
            for i, part in enumerate(partitions):
                output_path = self._output_dir.joinpath(f"{self._output_file_name}_part_{i + 1}")\
                                              .with_suffix(f".{self._get_extension()}")
                self._write_partition(dataset_part=part, output_path=output_path, part_id=str(i+1))
        else:
            for dataset in partitions:
                output_path = self._output_dir.joinpath(f"{self._output_file_name}")\
                                              .with_suffix(f".{self._get_extension()}")
                self._write_partition(dataset_part=dataset, output_path=output_path)

    def _write_partition(self, dataset_part, output_path: Path, part_id: str = None):
        examples_counter = 0
        for path, label in dataset_part:
            self._write_example(path, label)
            examples_counter += 1
        logging.info(f"{f'partition {part_id}: ' if part_id else ''}"
                     f"{examples_counter} examples have been written.")

    @abstractmethod
    def _write_example(self, path, label):
        ...

    @abstractmethod
    def _get_extension(self) -> str:
        ...
