import os

import dask.array as da
import dask.dataframe as dd
from dask_ml.linear_model import LinearRegression, LogisticRegression
from dask_ml.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import SGDRegressor, SGDClassifier

from ny_parking_violations_analysis import COLUMNS_DROP_TASK5_GROUPED_BY_DAY, COLUMNS_DROP_TASK5_CAR_MAKE
from ny_parking_violations_analysis.ml.eval import cr
from ny_parking_violations_analysis.ml.eval.metrics import get_regression_metrics
from ny_parking_violations_analysis.ml.eval.visualize import visualize
from ny_parking_violations_analysis.ml.ml_pipeline import (
    train_with_partial_fit_reg,
    train_with_dask_ml_reg,
    train_with_xgb,
    train_with_partial_fit_clf,
    train_with_dask_ml_clf,
    train_with_dummy_clf,
    train_with_dummy_reg
)
from ny_parking_violations_analysis.ml.transform_dataset import transform_for_training_day, transform_for_training_violation


def evaluate_violations_for_day(dataset: dd, reg_or_clf: str, alg: str, res_path: str):
    """Evaluate ML algorithms on tasks concerning violations on a particular day.

    The first task (regression) is predicting the number of violations given things we know about a particular day.

    The second task (classification) is predicting whether the number of violations will exceed the average.

    :param dataset: dataset to use
    :param reg_or_clf: which task to solve ('reg' for the first task, 'clf' for the second task.
    :param alg: algorithm to use (specified in the task instructions - 'partial_fit', 'dask_ml', 'xgb' or 'dummy')
    :param res_path: path to directory in which to store the results
    """

    if not os.path.isdir(res_path):
        raise ValueError('res_path must specify a directory')

    df_x, df_y = transform_for_training_day(dataset, COLUMNS_DROP_TASK5_GROUPED_BY_DAY, days_back_violations=3)

    if reg_or_clf == 'reg':
        # run task 1 regression problem

        x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, train_size=0.8, shuffle=True, random_state=0)

        if alg == 'partial_fit':
            reg_partial_fit = train_with_partial_fit_reg(x_train, y_train, reg=SGDRegressor())
            get_regression_metrics(y_test, reg_partial_fit.predict(x_test), res_path, 'reg_partial_fit_rm_1')
        elif alg == 'dask_ml':
            reg_dask_ml = train_with_dask_ml_reg(x_train, y_train, reg=LinearRegression())
            get_regression_metrics(y_test, reg_dask_ml.predict(x_test.values.compute_chunk_sizes()), res_path, 'reg_dask_ml_rm_1')
        elif alg == 'xgb':
            reg_xgb = train_with_xgb(x_train, y_train, reg_or_clf='reg')
            get_regression_metrics(y_test, reg_xgb.predict(x_test), res_path, 'reg_xgb_rm_1')
        elif alg == 'dummy':
            reg_dummy = train_with_dummy_reg(x_train, y_train)
            get_regression_metrics(y_test, reg_dummy.predict(x_test), res_path, 'reg_dummy_rm_1')
        else:
            raise NotImplementedError('{0} not supported'.format(alg))

    elif reg_or_clf == 'clf':
        # run task 2 classification problem

        _CLF_LABEL_DISPLAY_NAMES = ['below average', 'above average']
        _CLF_LABELS = [False, True]

        df_y = df_y > df_y.mean()
        x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, train_size=0.8, shuffle=True, random_state=0)

        if alg == 'partial_fit':
            clf_partial_fit = train_with_partial_fit_clf(x_train, y_train, clf=SGDClassifier(loss='modified_huber'), all_classes=list(df_y.unique().compute()))
            predictions = clf_partial_fit.predict(x_test)

            visualize.plot_confusion_matrix(predictions, y_test, _CLF_LABELS, _CLF_LABEL_DISPLAY_NAMES, res_path, 'clf_partial_fit_cm_1')
            visualize.plot_roc(clf_partial_fit.predict_proba(x_test), y_test, pos_label=True, plot_path=res_path, file_name_for_plot='clf_partial_fit_roc_1')
            cr_partial_fit = metrics.classification_report(y_test, clf_partial_fit.predict(x_test))
            cr.write_classification_report(cr_partial_fit, res_path, 'clf_partial_fit_cr_1')
        elif alg == 'dask_ml':
            clf_dask_ml = train_with_dask_ml_clf(x_train, y_train, clf=LogisticRegression())
            predictions = clf_dask_ml.predict(x_test.values.compute_chunk_sizes())

            visualize.plot_confusion_matrix(predictions, y_test, _CLF_LABELS, _CLF_LABEL_DISPLAY_NAMES, res_path, 'clf_dask_ml_cm_1')
            visualize.plot_roc(clf_dask_ml.predict_proba(x_test.values.compute_chunk_sizes()), y_test, pos_label=True, plot_path=res_path, file_name_for_plot='clf_dask_ml_roc_1')
            cr_dask_ml = metrics.classification_report(y_test, predictions)
            cr.write_classification_report(cr_dask_ml, res_path, 'clf_xgb_cr_1')
        elif alg == 'xgb':
            clf_xgb = train_with_xgb(x_train, y_train, reg_or_clf='clf_binary')
            predictions = clf_xgb.predict(x_test)

            visualize.plot_confusion_matrix(predictions, y_test, _CLF_LABELS, _CLF_LABEL_DISPLAY_NAMES, res_path, 'clf_xgb_cm_1')
            visualize.plot_roc(clf_xgb.predict_proba(x_test), y_test, pos_label=True, plot_path=res_path, file_name_for_plot='clf_xgb_roc_1')
            cr_xgb = metrics.classification_report(y_test, predictions)
            cr.write_classification_report(cr_xgb, res_path, 'clf_xgb_cr_1')
        elif alg == 'dummy':
            clf_dummy = train_with_dummy_clf(x_train, y_train)
            predictions = clf_dummy.predict(x_test)

            visualize.plot_confusion_matrix(predictions, y_test, _CLF_LABELS, _CLF_LABEL_DISPLAY_NAMES, res_path, 'clf_dummy_cm_1')
            visualize.plot_roc(clf_dummy.predict_proba(x_test), y_test, pos_label=True, plot_path=res_path, file_name_for_plot='clf_dummy_roc_1')
            cr_dummy = metrics.classification_report(y_test, predictions)
            cr.write_classification_report(cr_dummy, res_path, 'clf_dummy_cr_1')
    else:
        raise NotImplementedError('only \'reg\' or \'clf\' options are supported.')


def evaluate_car_make(dataset: dd, alg: str, res_path: str):
    """Evaluate ML algorithms on a vehicle make prediction task.

    :param dataset: dataset to use
    :param alg: algorithm to use (specified in the task instructions - 'partial_fit', 'dask_ml', 'xgb' or 'dummy')
    :param res_path: path to directory in which to store the results
    """

    if not os.path.isdir(res_path):
        raise ValueError('res_path must specify a directory')

    df_x, df_y, le = transform_for_training_violation(dataset, COLUMNS_DROP_TASK5_CAR_MAKE)

    # get unique car makes
    unique_car_makes = da.unique(df_y).compute()

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, convert_mixed_types=True, train_size=0.8, shuffle=True, random_state=0)
    if alg == 'partial_fit':
        clf_partial_fit = train_with_partial_fit_clf(x_train, y_train, clf=SGDClassifier(loss='modified_huber'), all_classes=unique_car_makes)
        cr_partial_fit = metrics.classification_report(y_test, clf_partial_fit.predict(x_test))
        cr.write_classification_report(cr_partial_fit, res_path, 'clf_partial_fit_cr_2')
    elif alg == 'xgb':
        clf_xgb = train_with_xgb(x_train, y_train, reg_or_clf='clf_multi', num_class=unique_car_makes.size)
        cr_xgb = metrics.classification_report(y_test, clf_xgb.predict(x_test))
        cr.write_classification_report(cr_xgb, res_path, 'clf_xgb_cr_2')
    elif alg == 'dummy':
        clf_dummy = train_with_dummy_clf(x_train, y_train)
        cr_dummy = metrics.classification_report(y_test, clf_dummy.predict(x_test))
        cr.write_classification_report(cr_dummy, res_path, 'clf_dummy_cr_2')
    else:
        raise NotImplementedError('{0} not supported'.format(alg))
