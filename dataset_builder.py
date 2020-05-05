import logging
from pathlib import Path
from typing import Tuple, List

from sklearn.model_selection import KFold, StratifiedKFold, BaseCrossValidator
import numpy as np

import utils


class DatasetBuilder:
    def __init__(self, data_root: Path, n_splits: int, shuffle: bool, stratify: bool, seed: int = None):
        self._data_root = data_root
        self._n_splits = n_splits
        self._shuffle = shuffle
        self._stratify = stratify
        self._seed = seed

    def build(self):
        paths, labels = self._extract_paths_and_labels()
        n_examples = len(paths)
        logging.info(f"{n_examples} examples have been found.")
        if n_examples > 0:
            return self._iter_dataset(paths, labels)

    def _extract_paths_and_labels(self) -> Tuple[List[Path], List[str]]:
        file_paths = []
        labels = []
        for class_folder in utils.iter_folders(self._data_root):
            label = class_folder.name
            for file_path in utils.iter_files(class_folder):
                if self.is_file_integral(file_path):
                    file_paths.append(file_path)
                    labels.append(label)
        return file_paths, labels

    def is_file_integral(self, file_path: Path) -> bool:
        return True

    def _iter_dataset(self, file_paths: List[Path], labels: List[str]):
        file_paths = np.array(file_paths)
        labels = np.array(labels)
        if self._n_splits > 1:
            splitter = self._create_splitter()
            for p, (_, indices) in enumerate(splitter.split(file_paths, labels)):
                yield self._iter_partition(file_paths[indices], labels[indices])
        else:
            np.random.seed(self._seed)
            indices = np.random.choice(np.arange(len(file_paths)), len(file_paths), replace=False)
            yield self._iter_partition(file_paths[indices], labels[indices])

    @staticmethod
    def _iter_partition(file_paths: np.array, labels: np.array):
        for path, label in zip(file_paths, labels):
            yield path, label

    def _create_splitter(self) -> BaseCrossValidator:
        if self._stratify:
            return StratifiedKFold(n_splits=self._n_splits, shuffle=self._shuffle, random_state=self._seed)
        else:
            return KFold(n_splits=self._n_splits, shuffle=self._shuffle, random_state=self._seed)
