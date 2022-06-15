import os
import re

import dask.dataframe as dd
import pandas as pd
import requests
import tqdm
from bs4 import BeautifulSoup, NavigableString

from ny_parking_violations_analysis import get_google_api_key
from ny_parking_violations_analysis.data_augmentation import PATH_TO_CACHED_ATTRACTIONS, PATH_TO_CACHED_ATTRACTIONS_DISTS
from ny_parking_violations_analysis.data_augmentation.specific import WEBSITE_ATTRACTIONS_PATH
from ny_parking_violations_analysis.data_augmentation.specific import get_dist_closest
from ny_parking_violations_analysis.data_augmentation.specific import name_to_coordinates

from . import logger

def join_with(df: dd, street_coordinates: dict) -> dd:
    """Join dataframe representing the main dataset with the dataframe containing the distance to the closest major
    attraction and the name of the attraction (as stated on the specified website).

    :param df: dataframe representing the main dataset
    :param street_coordinates: mapping of unique streets in the main dataset to their coordinates
    :return: augmented dataset
    """

    # parse cached DataFrame for attractions if exists, else compute new and cache
    if os.path.exists(PATH_TO_CACHED_ATTRACTIONS):
        logger.info('Cached distances to nearest attractions exist')

        df_attractions = pd.read_pickle(PATH_TO_CACHED_ATTRACTIONS)
    else:
        logger.info('Computing distances to nearest attractions exist')

        df_attractions = parse_attractions()
        df_attractions.to_pickle(PATH_TO_CACHED_ATTRACTIONS)

    # parse cached DataFrame for nearest attractions if exists, else compute new and cache
    if os.path.exists(PATH_TO_CACHED_ATTRACTIONS_DISTS):
        closest_attractions_dists = pd.read_pickle(PATH_TO_CACHED_ATTRACTIONS_DISTS)
    else:
        street_and_closest_attraction = []
        for street, coordinates in tqdm.tqdm(street_coordinates.items(), desc='computing closest attraction for specified streets', unit='street'):
            street_and_closest_attraction.append((street, *get_dist_closest(coordinates, df_attractions)))

        closest_attractions_dists = pd.DataFrame(street_and_closest_attraction, columns=['Street Name', 'nearest_attraction_dist', 'nearest_attraction_name'])
        closest_attractions_dists.to_pickle(PATH_TO_CACHED_ATTRACTIONS_DISTS)

    return dd.merge(df, closest_attractions_dists, on='Street Name', how='left')


def parse_attractions() -> pd.DataFrame:
    """Parse and geocode list of attractions in NYC from specified website.

    :return: Pandas DataFrame containing attraction names and their coordinates (['NAME', 'LATITUDE', 'LONGITUDE'])
    """

    # get list of attraction names
    soup = BeautifulSoup(requests.get(WEBSITE_ATTRACTIONS_PATH).text, 'lxml')
    h2_tags = soup.find_all('h2')
    patt_attractions = re.compile('[0-9]+ –')
    attraction_strs = [tag.contents[0].replace(u'\xa0', ' ') for tag in h2_tags
                       if isinstance(tag.contents[0], NavigableString) and patt_attractions.match(tag.contents[0])]
    attraction_names = [a[a.find('–') + 2:] for a in attraction_strs]

    # geocode attractions and convert to Pandas DataFrame
    geocoded_attractions = []
    google_api_key = get_google_api_key()
    for attraction_name in tqdm.tqdm(attraction_names, desc='Performing geocoding of parsed attraction names using the Google Maps API', unit='attraction name'):
        geocoded_attractions.append((attraction_name, name_to_coordinates(attraction_name, 'NY', google_api_key)))

    return pd.DataFrame([[el[0], el[1]['lat'], el[1]['lng']] for el in geocoded_attractions], columns=['NAME', 'LATITUDE', 'LONGITUDE'])
