from typing import Iterable

import dask.dataframe
import dask.dataframe as dd
import numpy as np
import xgboost as xgb
from sklearn.dummy import DummyClassifier, DummyRegressor


def train_with_partial_fit_reg(x_train: dd, y_train: dask.dataframe.Series, reg):
    for x_train_partition, y_train_partition in zip(x_train.partitions, y_train.partitions):
        reg.partial_fit(x_train_partition, y_train_partition)
    return reg


def train_with_partial_fit_clf(x_train: dd, y_train: dask.dataframe.Series, clf, all_classes: Iterable):
    for x_train_partition, y_train_partition in zip(x_train.partitions, y_train.partitions):
        clf.partial_fit(x_train_partition, y_train_partition, classes=all_classes)
    return clf


def train_with_dask_ml_reg(x_train: dd, y_train: dask.dataframe.Series, reg):
    return reg.fit(x_train.values, y_train.values)


def train_with_dask_ml_clf(x_train: dd, y_train: dask.dataframe.Series, clf):
    return clf.fit(x_train.values, y_train.values)


def train_with_xgb(client, x_train: dd, y_train: dask.dataframe.Series, reg_or_clf='reg', num_class=2):
    dtrain = xgb.dask.DaskDMatrix(client, x_train, y_train)

    if reg_or_clf == 'reg':
        objective = 'reg:squarederror'
    elif reg_or_clf == 'clf_binary':
        objective = 'binary:logistic'
    elif reg_or_clf == 'clf_multi':
        objective = 'multi:softmax'
    else:
        raise NotImplementedError('{0} not supported.'.format(reg_or_clf))

    param_dict = {
        "tree_method": "hist",
        "objective": objective,
    }

    if reg_or_clf == 'clf_multi':
        param_dict['num_class'] = num_class

    output = xgb.dask.train(
        client,
        param_dict,
        dtrain,
        evals=[(dtrain, "train")],
    )

    class Clf:

        def __init__(self, predict):
            if reg_or_clf == 'reg':
                self.predict = predict
            elif reg_or_clf == 'clf_binary':
                def predict_proba(x):
                    pred = predict(x)
                    return np.array([[1 - p, p] for p in pred])

                self.predict_proba = predict_proba
                self.predict = lambda x: predict(x) > 0.5
            elif reg_or_clf == 'clf_multi':
                self.predict = lambda x: predict(x).astype(int)
            else:
                raise NotImplementedError('{0} not supported.'.format(reg_or_clf))

    return Clf(predict=lambda x: xgb.dask.predict(client, output, x).compute())


def train_with_dummy_reg(x_train: dd, y_train: dask.dataframe.Series, **kwargs):
    reg = DummyRegressor(**kwargs)
    reg.fit(None, y_train)
    return reg


def train_with_dummy_clf(x_train: dd, y_train: dask.dataframe.Series, **kwargs):
    clf = DummyClassifier(**kwargs)
    clf.fit(None, y_train)
    return clf
