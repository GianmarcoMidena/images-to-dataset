from collections import defaultdict

import numpy as np
import pandas as pd


class StratifiedGroupKFold:
    _X_COL = 'X'
    _Y_COL = 'y'
    _GROUP_COL = 'group'

    def __init__(self, n_splits: int):
        self._n_splits = n_splits

    def split(self, X: np.array, y: np.array, groups: np.array):
        data = pd.DataFrame(np.vstack([X, y, groups]).T, columns=[self._X_COL, self._Y_COL, self._GROUP_COL])
        n_classes = len(data[self._Y_COL].unique())
        labels_count = data[self._Y_COL].value_counts()
        max_labels_count_per_fold = labels_count / self._n_splits

        labels_count_by_fold = pd.DataFrame(np.zeros([self._n_splits, n_classes]), columns=data[self._Y_COL].unique())
        labels_count_by_group = data.assign(value=1)\
                                    .pivot_table(index=self._GROUP_COL, columns=self._Y_COL,
                                                 values='value', aggfunc="sum", fill_value=0)

        labels_count_by_group = self._sort_by_descending_row_std(labels_count_by_group)
        groups_per_fold = defaultdict(set)

        for group, group_labels_count in labels_count_by_group.iterrows():
            best_fold_for_group, min_eval_for_group = \
                self._find_best_fold_for_group(group_labels_count=group_labels_count,
                                               labels_count_by_fold=labels_count_by_fold, labels_count=labels_count,
                                               max_labels_count_per_fold=max_labels_count_per_fold)

            labels_count_by_fold.loc[best_fold_for_group] += group_labels_count
            groups_per_fold[best_fold_for_group].add(group)

        yield from self._generate_indices(groups_per_fold, data)

    @staticmethod
    def _sort_by_descending_row_std(labels_count_by_group):
        labels_count_by_group = labels_count_by_group.assign(row_std=labels_count_by_group.std(axis=1)) \
                                                     .sort_values(by='row_std', ascending=False) \
                                                     .drop(columns='row_std')
        return labels_count_by_group

    def _find_best_fold_for_group(self, group_labels_count: pd.Series, labels_count_by_fold: pd.DataFrame,
                                  labels_count: pd.Series, max_labels_count_per_fold: pd.Series):
        best_fold_for_group = None
        min_eval = None
        for fold_i in self._search_folds(labels_count_by_fold, max_labels_count_per_fold):
            fold_eval = self._eval_label_counts_per_fold(fold_i=fold_i, group_labels_count=group_labels_count,
                                                         labels_count_by_fold=labels_count_by_fold,
                                                         labels_count=labels_count)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold_for_group = fold_i
        return best_fold_for_group, min_eval

    def _search_folds(self, labels_count_by_fold, max_labels_count_per_fold):
        folds = labels_count_by_fold.loc[
            labels_count_by_fold.lt(max_labels_count_per_fold, axis=1).all(axis=1)].index.tolist()
        if len(folds) == 0:
            tot_labels_count_by_fold = labels_count_by_fold.sum(axis=1)
            folds = tot_labels_count_by_fold[tot_labels_count_by_fold == tot_labels_count_by_fold.min()].index.tolist()
        return folds

    def _eval_label_counts_per_fold(self, fold_i: int, group_labels_count: pd.Series,
                                    labels_count_by_fold: pd.DataFrame, labels_count: pd.Series):
        labels_count_by_fold = labels_count_by_fold.copy()
        labels_count_by_fold.loc[fold_i] += group_labels_count

        std_mean = self._std_mean(labels_count_by_fold, labels_count)
        labels_count_by_fold.loc[fold_i] -= group_labels_count
        return std_mean

    def _std_mean(self, labels_count_by_fold: pd.DataFrame, labels_count: pd.Series):
        return labels_count_by_fold.div(labels_count, axis=1) \
                                   .std(axis=1) \
                                   .mean()

    def _generate_indices(self, groups_per_fold, data):
        all_groups = set(data[self._GROUP_COL].unique())
        for fold in range(self._n_splits):
            yield from self.generate_indices_by_fold(fold, all_groups, groups_per_fold, data)

    def generate_indices_by_fold(self, fold, all_groups, groups_per_fold, data):
        train_groups = all_groups - groups_per_fold[fold]
        test_groups = groups_per_fold[fold]
        train_indices = data[data[self._GROUP_COL].isin(train_groups)].index.tolist()
        test_indices = data[data[self._GROUP_COL].isin(test_groups)].index.tolist()
        yield train_indices, test_indices
