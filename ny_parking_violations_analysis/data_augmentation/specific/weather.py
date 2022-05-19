import os

import dask.dataframe as dd
import pandas as pd


def join_with(df: dd) -> dd:
    """Join dataframe representing the main dataset with weather information for each particular day.

    :param df: dataframe representing the main dataset
    :return: augmented dataset
    """

    # Parse weather data and join to augment main dataset
    df_weather = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/nyc_weather_2021_2022.csv'), delimiter=',')
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])

    return dd.merge(df, df_weather.rename(columns={'datetime': 'Issue Date'}), on='Issue Date', how='left')
