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
    def __init__(self, n_splits: int, with_shuffle: bool, with_stratify: bool, group: str, path_column: str = 'path',
                 data_root: Path = None, metadata: Path = None, sequence_or_grid: bool = False, sample_id: str = None,
                 additional_columns: List[str] = None, label_column: str = 'label',
                 global_label_column: str = 'global_label', local_label_column: str = 'local_label', seed: int = None):
        self._data_root = data_root
        self._n_splits = n_splits
        self._with_shuffle = with_shuffle
        self._with_stratify = with_stratify
        self._group = group
        self._metadata = metadata
        self._path_column = path_column
        self._label_column = label_column
        self._global_label_column = global_label_column
        self._local_label_column = local_label_column
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

                path_existence = metadata[self._path_column].apply(Path.exists)
                metadata = metadata.loc[path_existence]
                examples = metadata[[self._path_column]]

                if self._sample_id:
                    examples[self._sample_id] = metadata[self._sample_id]

                if self._additional_columns and (len(self._additional_columns) > 0):
                    examples[self._additional_columns] = metadata[self._additional_columns]

                if not {self._label_column, self._local_label_column, self._global_label_column}\
                        .isdisjoint(metadata.columns):
                    if {self._local_label_column, self._global_label_column}.isdisjoint(metadata.columns):
                        metadata[self._label_column] = metadata[self._label_column].apply(str)
                        examples[self._label_column] = metadata[self._label_column]
                    else:
                        if self._local_label_column in metadata:
                            metadata[self._local_label_column] = metadata[self._local_label_column].apply(str)
                            examples[self._local_label_column] = metadata[self._local_label_column]

                        if self._global_label_column in metadata:
                            metadata[self._global_label_column] = metadata[self._global_label_column].apply(str)
                            examples[self._global_label_column] = metadata[self._global_label_column]

        if examples is None and self._data_root:
            examples = pd.DataFrame()
            for class_folder in utils.iter_folders(self._data_root):
                label = class_folder.name
                for file_path in utils.iter_files(class_folder):
                    if self.check_data_integrity(file_path):
                        examples = examples.append(
                            {self._path_column: file_path, self._label_column: label}, ignore_index=True, sort=False)

        if examples is not None:
            if self._sequence_or_grid:
                agg = {self._path_column: list}

                if self._global_label_column in examples:
                    agg[self._global_label_column] = self._mode
                elif self._label_column in examples:
                    agg[self._label_column] = self._mode

                if self._local_label_column in examples:
                    agg[self._local_label_column] = list

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

    def check_data_integrity(self, file_path: Path) -> bool:
        return True

    def _iter_dataset(self, examples: pd.DataFrame, groups: List[int] = None):
        file_paths = examples[self._path_column].values
        if groups is not None:
            groups = np.array(groups)
        if self._n_splits > 1:
            for p, (_, indices) in enumerate(self._split(examples, groups)):
                yield self._iter_partition(examples.iloc[indices])
        else:
            np.random.seed(self._seed)
            indices = np.random.choice(np.arange(len(file_paths)), len(file_paths), replace=False)
            yield self._iter_partition(examples.iloc[indices])

    def _iter_partition(self, examples: pd.DataFrame):
        for _, example in examples.iterrows():

            kwargs = {}

            if self._label_column in example:
                kwargs['label'] = example[self._label_column]
            else:
                if self._global_label_column in example:
                    kwargs['global_label'] = example[self._global_label_column]

                if self._local_label_column in example:
                    kwargs['local_label'] = example[self._local_label_column]

            if self._additional_columns:
                kwargs.update(example[self._additional_columns].to_dict())

            yield example['id'], example[self._path_column], kwargs

    def _split(self, examples: pd.DataFrame, groups: np.array = None):
        file_paths = examples[self._path_column].values
        if self._with_stratify and not {self._label_column, self._global_label_column}.isdisjoint(examples.columns):
            if self._global_label_column in examples:
                labels = examples[self._global_label_column].values
            else:
                labels = examples[self._label_column].values

            if self._group and groups is not None:
                return StratifiedGroupKFold(n_splits=self._n_splits) \
                    .split(file_paths, labels, groups)
            else:
                return StratifiedKFold(n_splits=self._n_splits, shuffle=self._with_shuffle, random_state=self._seed) \
                    .split(file_paths, labels)
        else:
            if self._group and groups is not None:
                return GroupKFold(n_splits=self._n_splits) \
                    .split(file_paths, groups)
            else:
                return KFold(n_splits=self._n_splits, shuffle=self._with_shuffle, random_state=self._seed) \
                    .split(file_paths)

    @staticmethod
    def _mode(values):
        return scipy.stats.mode(values)[0]
