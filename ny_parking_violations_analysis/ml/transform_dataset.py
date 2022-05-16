from typing import Iterable

import dask.array as da
import dask.dataframe as dd
import numpy as np


def transform_for_training_violation():
    """TODO

    :return:
    """
    pass


def transform_for_training_day(df: dd, columns_for_violation: Iterable[str], days_back_violations) -> dd:
    """Transform dataset that contains data for a particular day. The transformed dataset does not
    contain information that is known after processing the parking violations.

    :param df: dataset to process
    :param columns_for_violation: columns containing data obtained from parking violations
    :return: processed dataset
    """

    df_grouped = df.groupby('Issue Date').first().reset_index()
    df_grouped['month'] = df_grouped['Issue Date'].map(lambda x: x.month)
    df_grouped.sort_values(by='Issue Date')

    # add columns for violations for previous days
    violations_per_day = df.groupby('Issue Date').size().compute()
    df_grouped['num_violations'] = da.from_array(violations_per_day.to_numpy())
    for days_back in range(1, days_back_violations + 1):
        violations_per_day_shifted1 = da.from_array(np.roll(violations_per_day, days_back))
        df_grouped['violations_prev_days_{0}'.format(str(days_back))] = violations_per_day_shifted1

    # drop columns for data that is only known at time of violation
    return df_grouped.drop(columns_for_violation, axis=1)
