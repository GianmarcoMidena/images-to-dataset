import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

from img2ds.building import utils
from img2ds.splitting import StratifiedGroupKFold


class DatasetBuilder:
    def __init__(self, n_splits: int, with_shuffle: bool, with_stratify: bool, group: str,
                 path_column: str = 'path', label_column: str = 'label', data_root: Path = None, metadata: Path = None,
                 sequence_or_grid: bool = False, sample_id: str = None, additional_columns: List[str] = None,
                 seed: int = None):
        self._data_root = data_root
        self._n_splits = n_splits
        self._with_shuffle = with_shuffle
        self._with_stratify = with_stratify
        self._group = group
        self._metadata = metadata
        self._path_column = path_column
        self._label_column = label_column
        self._sequence_or_grid = sequence_or_grid
        self._sample_id = sample_id
        self._additional_columns = additional_columns
        self._seed = seed

    def build(self):
        examples, groups = self._extract_examples()
        n_examples = examples.shape[0]
        logging.info(f"{n_examples} examples have been found.")
        if n_examples > 0:
            return self._iter_dataset(examples, groups)
        return []

    def _extract_examples(self) -> Tuple[pd.DataFrame, List[int]]:
        metadata = None
        examples = None
        groups = []

        if self._metadata is not None:
            metadata = pd.read_csv(self._metadata)
            if self._path_column in metadata:
                metadata[self._path_column] = metadata[self._path_column].apply(Path)
                if self._label_column in metadata:
                    path_existence = metadata[self._path_column].apply(Path.exists)
                    metadata = metadata.loc[path_existence]
                    metadata[self._label_column] = metadata[self._label_column].apply(str)
                    examples = metadata[[self._path_column, self._label_column]]
                    if self._sample_id:
                        examples[self._sample_id] = metadata[self._sample_id]
                    if self._additional_columns and (len(self._additional_columns) > 0):
                        examples[self._additional_columns] = metadata[self._additional_columns]

        if examples is None and self._data_root:
            examples = pd.DataFrame()
            for class_folder in utils.iter_folders(self._data_root):
                label = class_folder.name
                for file_path in utils.iter_files(class_folder):
                    if self.is_file_integral(file_path):
                        examples = examples.append(
                            {self._path_column: file_path, self._label_column: label}, ignore_index=True, sort=False)

        if examples is not None:
            if self._sequence_or_grid:
                agg = {self._path_column: list, self._label_column: self._mode}
                if self._group and self._group in examples:
                    agg[self._group] = self._mode
                examples = examples.groupby(self._sample_id, sort=False).agg(agg)
                examples = examples.reset_index(drop=False).rename(columns={self._sample_id: 'id'})
            elif self._sample_id:
                examples = examples.rename(columns={self._sample_id: 'id'})
            else:
                examples['id'] = examples[self._path_column].apply(str)\
                                                .str.rsplit('/', n=1, expand=True)[1]\
                                                .str.rsplit('.', n=1, expand=True)[0]
            if self._with_shuffle:
                examples = examples.sample(frac=1, replace=False, axis=0, random_state=self._seed)
            if self._group and metadata is not None and \
                    all(c in metadata.columns for c in [self._path_column, self._group]):
                groups = metadata.set_index(self._path_column) \
                    .loc[examples[self._path_column].values, self._group]

        return examples, groups

    def is_file_integral(self, file_path: Path) -> bool:
        return True

    def _iter_dataset(self, examples: pd.DataFrame, groups: List[int] = None):
        file_paths = examples[self._path_column].values
        labels = examples[self._label_column].values
        if groups is not None:
            groups = np.array(groups)
        if self._n_splits > 1:
            for p, (_, indices) in enumerate(self._split(file_paths, labels, groups)):
                yield self._iter_partition(examples.iloc[indices])
        else:
            np.random.seed(self._seed)
            indices = np.random.choice(np.arange(len(file_paths)), len(file_paths), replace=False)
            yield self._iter_partition(examples.iloc[indices])

    def _iter_partition(self, examples: pd.DataFrame):
        for _, example in examples.iterrows():
            if self._additional_columns:
                kwargs = example[self._additional_columns].to_dict()
            else:
                kwargs = {}
            yield example['id'], example[self._path_column], example[self._label_column], kwargs

    def _split(self, file_paths: np.array, labels: np.array, groups: np.array = None):
        if self._with_stratify:
            if self._group and groups is not None:
                return StratifiedGroupKFold(n_splits=self._n_splits) \
                    .split(file_paths, labels, groups)
            else:
                return StratifiedKFold(n_splits=self._n_splits, shuffle=self._with_shuffle, random_state=self._seed) \
                    .split(file_paths, labels)
        else:
            if self._group and groups is not None:
                return GroupKFold(n_splits=self._n_splits) \
                    .split(file_paths, labels, groups)
            else:
                return KFold(n_splits=self._n_splits, shuffle=self._with_shuffle, random_state=self._seed) \
                    .split(file_paths, labels)

    @staticmethod
    def _mode(values):
        return scipy.stats.mode(values)[0]
