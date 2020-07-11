from pathlib import Path
import pandas as pd

from img2ds.writing import SimpleDatasetWriter


class CSVWriter(SimpleDatasetWriter):
    def _write_partition(self, dataset_part, output_path: Path, part_id: str = None):
        self._data = pd.DataFrame(columns=['path', 'label'])
        super()._write_partition(dataset_part=dataset_part, output_path=output_path, part_id=part_id)
        self._data.to_csv(output_path, index=False)
        self._data = None

    def _write_example(self, id: str, path: Path, label: str, **kwargs):
        self._data = self._data.append({'id': id, 'path': path, 'label': label, **kwargs},
                                       ignore_index=True, sort=False)

    def _get_extension(self) -> str:
        return "csv"
