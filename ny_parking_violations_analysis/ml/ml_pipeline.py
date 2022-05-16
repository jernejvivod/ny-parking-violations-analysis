from typing import Iterable

import dask.dataframe
import dask.dataframe as dd


def get_ml_pipeline(df: dd):
    # TODO
    pass


def train_with_partial_fit(x_train: dd, y_train: dask.dataframe.Series, clf, all_classes: Iterable):
    for x_train_partition, y_train_partition in zip(x_train.partitions, y_train.partitions):
        clf.partial_fit(x_train_partition, y_train_partition, classes=all_classes)


def train_with_dask_ml():
    # TODO
    pass
