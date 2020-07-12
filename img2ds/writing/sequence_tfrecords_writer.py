from abc import abstractmethod
from pathlib import Path

from typing import List

from img2ds.writing import TFRecordsWriter, SequenceDatasetWriter


class SequenceTFRecordsWriter(TFRecordsWriter, SequenceDatasetWriter):
    def _write_example(self, id: str, paths: List[Path], **kwargs):
        example = self._make_example(id, paths, **kwargs)
        self._writer.write(example)

    @abstractmethod
    def _make_example(self, id: str, paths: List[Path], **kwargs):
        ...
