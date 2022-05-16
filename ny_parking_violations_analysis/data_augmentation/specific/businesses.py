import os
import re

import dask.dataframe as dd
import pandas as pd
import requests
import tqdm
from bs4 import BeautifulSoup, NavigableString

from ny_parking_violations_analysis import get_google_api_key
from ny_parking_violations_analysis.data_augmentation import PATH_TO_CACHED_BUSINESSES, PATH_TO_CACHED_BUSINESSES_DISTS
from ny_parking_violations_analysis.data_augmentation.specific import WEBSITE_BUSINESSES_PATH
from ny_parking_violations_analysis.data_augmentation.specific import get_dist_closest
from ny_parking_violations_analysis.data_augmentation.specific import name_to_coordinates


def join_with(df: dd, street_coordinates: dict) -> dd:
    """Join dataframe representing the main dataset with the dataframe containing the distance to the closest major
    business and the name of the business (as stated on the specified website).

    :param df: dataframe representing the main dataset
    :param street_coordinates: mapping of unique streets in the main dataset to their coordinates
    :return: augmented dataset
    """

    # parse cached DataFrame for businesses if exists, else compute new and cache
    if os.path.exists(PATH_TO_CACHED_BUSINESSES):
        df_businesses = pd.read_pickle(PATH_TO_CACHED_BUSINESSES)
    else:
        df_businesses = parse_businesses()
        df_businesses.to_pickle(PATH_TO_CACHED_BUSINESSES)

    # parse cached DataFrame for nearest businesses if exists, else compute new and cache
    if os.path.exists(PATH_TO_CACHED_BUSINESSES_DISTS):
        closest_businesses_dists = pd.read_pickle(PATH_TO_CACHED_BUSINESSES_DISTS)
    else:
        street_and_closest_business = []
        for street, coordinates in tqdm.tqdm(street_coordinates.items(), desc='computing closest businesses for specified streets', unit='street'):
            street_and_closest_business.append((street, *get_dist_closest(coordinates, df_businesses)))

        closest_businesses_dists = pd.DataFrame(street_and_closest_business, columns=['Street Name', 'nearest_business_dist', 'nearest_business_name'])
        closest_businesses_dists.to_pickle(PATH_TO_CACHED_BUSINESSES_DISTS)

    return dd.merge(df, closest_businesses_dists, on='Street Name')


def parse_businesses():
    """Parse and geocode list of major businesses in NYC from specified website.

    :return: Pandas DataFrame containing business names and their coordinates (['NAME', 'LATITUDE', 'LONGITUDE'])
    """

    soup = BeautifulSoup(requests.get(WEBSITE_BUSINESSES_PATH).text, 'lxml')
    patt_business = re.compile('#[0-9]+')
    business_strs = [el.contents[0] for el in soup.find_all('b')
                     if isinstance(el.contents[0], NavigableString) and patt_business.match(el.contents[0])]
    patt_extract_name = re.compile('#[0-9]+ (.*) \(')
    business_names = [patt_extract_name.match(el).group(1) for el in business_strs]

    # geocode businesses and convert to Pandas DataFrame
    geocoded_businesses = []
    google_api_key = get_google_api_key()
    for attraction_name in tqdm.tqdm(business_names, desc='Performing geocoding of parsed business names using the Google Maps API', unit='name'):
        geocoded_businesses.append((attraction_name, name_to_coordinates(attraction_name, 'NY', google_api_key)))

    return pd.DataFrame([[el[0], el[1]['lat'], el[1]['lng']] for el in geocoded_businesses], columns=['NAME', 'LATITUDE', 'LONGITUDE'])
