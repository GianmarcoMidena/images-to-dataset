from abc import abstractmethod
from pathlib import Path

from img2ds.writing import TFRecordsWriter, SimpleDatasetWriter


class SimpleTFRecordsWriter(TFRecordsWriter, SimpleDatasetWriter):
    def _write_example(self, id: str, path: Path, label: str, **kwargs):
        example = self._make_example(id, path, label, **kwargs)
        self._writer.write(example)

    @abstractmethod
    def _make_example(self, id: str, path: Path, label: str, **kwargs):
        ...
