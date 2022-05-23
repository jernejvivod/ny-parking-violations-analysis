from typing import Iterable

import dask.array as da
import dask.dataframe as dd
import numpy as np
from dask_ml.impute import SimpleImputer
from dask_ml.preprocessing import DummyEncoder, RobustScaler, LabelEncoder


def transform_for_training_violation(df: dd, columns_to_drop: Iterable[str]):
    """Transform dataset for per-violation ML tasks.

    :param df: dataset to process
    :param columns_to_drop: columns to drop
    :return: processed dataset (samples and labels)
    """

    # process dataset
    df = df[df['Vehicle Make'].notnull()]
    df_x, df_y = df.loc[:, df.columns != 'Vehicle Make'], df['Vehicle Make']
    df_ohe = DummyEncoder().fit_transform(df_x.drop(columns_to_drop, axis=1).categorize())
    le = LabelEncoder()
    return SimpleImputer().fit_transform(df_ohe), le.fit_transform(df_y.map(lambda x: x.strip())), le


def transform_for_training_day(df: dd, columns_to_drop: Iterable[str], days_back_violations) -> dd:
    """Transform dataset to a dataset that contains data for a particular day. The transformed dataset does not
    contain information that is known after processing the parking violations.

    :param df: dataset to process
    :param columns_to_drop: columns to drop (should include columns obtained from parking violations)
    :param days_back_violations: how many columns specifying the number of violations in the previous k days to add
    :return: processed dataset (samples and labels)
    """

    df_grouped = df.drop(list(filter(lambda x: x != 'Issue Date', columns_to_drop)), axis=1).groupby('Issue Date').first().reset_index()
    df_grouped['month'] = df_grouped['Issue Date'].map(lambda x: x.month)
    df_grouped.sort_values(by='Issue Date')

    # add columns for violations for previous days
    violations_per_day = df.groupby('Issue Date').size().compute()
    df_grouped['num_violations'] = da.from_array(violations_per_day.to_numpy())
    for days_back in range(1, days_back_violations + 1):
        violations_per_day_shifted1 = da.from_array(np.roll(violations_per_day, days_back))
        df_grouped['violations_prev_days_{0}'.format(str(days_back))] = violations_per_day_shifted1

    # drop columns for data that is only known at time of violation
    df_grouped = df_grouped.drop('Issue Date', axis=1)
    for col in df_grouped.select_dtypes(include=[object]):
        df_grouped[col] = df_grouped[col].astype('category')

    df_x, df_y = df_grouped.loc[:, df_grouped.columns != 'num_violations'], df_grouped['num_violations']

    df_grouped_ohe = DummyEncoder().fit_transform(df_x.categorize())
    df_grouped_ohe_imputed = SimpleImputer().fit_transform(df_grouped_ohe)
    return RobustScaler().fit_transform(df_grouped_ohe_imputed).persist(), df_y.persist()
