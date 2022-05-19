import os
import time
from datetime import datetime

import dask.dataframe as dd
import pandas as pd
import tqdm
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from undetected_chromedriver import Chrome

from ny_parking_violations_analysis import get_google_api_key
from ny_parking_violations_analysis.data_augmentation import PATH_TO_CACHED_CONCERTS, PATH_TO_CACHED_GEOCODED_CONCERTS
from ny_parking_violations_analysis.data_augmentation.specific import WEBSITE_CONCERTS_PATH_TEMPLATE
from ny_parking_violations_analysis.data_augmentation.specific import name_to_coordinates


def join_with(df: dd) -> dd:
    """Join dataframe representing the main dataset with the dataframe containing the number of major concerts for that day
    (as stated on the specified website).

    :param df: dataframe representing the main dataset
    :return: augmented dataset
    """

    # parse cached DataFrame for concerts if exists, else compute new and cache
    if os.path.exists(PATH_TO_CACHED_GEOCODED_CONCERTS):
        df_concerts_geocoded_cleaned = pd.read_pickle(PATH_TO_CACHED_GEOCODED_CONCERTS)
    else:
        # parse cached DataFrame for raw concert data if exists, else compute new and cache
        if os.path.exists(PATH_TO_CACHED_CONCERTS):
            df_concerts = pd.read_pickle(PATH_TO_CACHED_CONCERTS)
        else:
            df_concerts = parse_concerts(year_limit=2019)
            df_concerts.to_pickle(PATH_TO_CACHED_CONCERTS)

        df_relevant_cols = df_concerts[['Date', 'Concert', 'Venue']]
        unique_venues = df_relevant_cols['Venue'].unique()

        # geocode unique venues and prepare DataFrame for Augmentation
        geocoded_venues_list = []
        google_api_key = get_google_api_key()
        for venue_name in tqdm.tqdm(unique_venues, desc='Performing geocoding of parsed venues using the Google Maps API', unit='name'):
            if isinstance(venue_name, str):
                geocoded_venues_list.append((venue_name, name_to_coordinates(venue_name, 'NY', google_api_key)))

        geocoded_venues_df = pd.DataFrame([[el[0], el[1]['lat'], el[1]['lng']] for el in geocoded_venues_list], columns=['Venue', 'LATITUDE', 'LONGITUDE'])
        merged_df = pd.merge(df_relevant_cols, geocoded_venues_df, on='Venue')
        df_concerts_geocoded_cleaned = merged_df.drop_duplicates(['Date', 'LATITUDE', 'LONGITUDE'])
        df_concerts_geocoded_cleaned.to_pickle(PATH_TO_CACHED_GEOCODED_CONCERTS)

    # compute number of concerts for each relevant date and augment dataset
    num_concerts_on_date = df_concerts_geocoded_cleaned.groupby('Date').size().reset_index(name='num_concerts').rename(columns={'Date': 'Issue Date'})
    num_concerts_on_date = num_concerts_on_date[num_concerts_on_date['Issue Date'] != datetime.min]
    num_concerts_on_date['Issue Date'] = pd.to_datetime(num_concerts_on_date['Issue Date'])

    df_merged = dd.merge(df, num_concerts_on_date, on='Issue Date', how='left')
    df_merged['num_concerts'] = df_merged['num_concerts'].fillna(0)
    return df_merged


def parse_concerts(year_limit):
    """Parse information about major concerts in NYC from specified website.

    :return: Pandas DataFrame containing information about major concerts in NYC from specified website
    """

    # initialize driver
    driver = Chrome()

    # parse tables
    dfs_for_pages = []
    end = False
    idx_page = 1
    while not end:

        # try to parse table and convert to DataFrame
        try:
            driver.get(WEBSITE_CONCERTS_PATH_TEMPLATE.format(idx_page))
            time.sleep(1)
            df = pd.read_html(driver.find_element_by_id('band-show-table').get_attribute('outerHTML'))[0]
        except (NoSuchElementException, WebDriverException):
            continue
        df['Date'] = df['Date'].map(parse_table_date)
        idx_page += 1

        # stop parsing when reached end of range
        if df['Date'].iloc[-1].year == year_limit:
            end = True
        else:
            dfs_for_pages.append(df)

    # return concatenated DataFrames for pages
    return pd.concat(dfs_for_pages)


def parse_table_date(date_str: str):
    for fmt in ('%b %d, %Y', '%b %d %Y'):
        try:
            res = datetime.strptime(''.join((ch for ch in date_str if ch.isalnum() or ch.isspace())), fmt)
            return res
        except ValueError:
            pass
    return datetime.min
