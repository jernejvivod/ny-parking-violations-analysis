import os

import dask.dataframe as dd
import pandas as pd
import tqdm

from ny_parking_violations_analysis.data_augmentation import PATH_TO_CACHED_SCHOOL_DISTS
from ny_parking_violations_analysis.data_augmentation.specific import get_dist_closest
from . import logger


def join_with(df: dd, street_coordinates: dict) -> dd:
    """Join dataframe representing the main dataset with the dataframe containing the distance to the closest elementary or high school
    and the name of the school (using the provided dataset).

    :param df: dataframe representing the main dataset
    :param street_coordinates: mapping of unique streets in the main dataset to their coordinates
    :return: augmented dataset
    """

    # parse cached DataFrame for nearest schools if exists, else compute new and cache
    if os.path.exists(PATH_TO_CACHED_SCHOOL_DISTS):
        logger.info('Cached distances to nearest schools exist')
        closest_school_dists = pd.read_pickle(PATH_TO_CACHED_SCHOOL_DISTS)
    else:
        logger.info('Computing distances to nearest schools')

        # parse dataset and compute nearest school for each provided street
        df_schools = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/2019_-_2020_School_Locations.csv'), delimiter=',')
        street_and_closest_school = []
        for street, coordinates in tqdm.tqdm(street_coordinates.items(), desc='computing closest schools for specified streets', unit='street'):
            street_and_closest_school.append((street, *get_dist_closest(coordinates, df_schools, name_col='location_name')))

        closest_school_dists = pd.DataFrame(street_and_closest_school, columns=['Street Name', 'nearest_school_dist', 'nearest_school_name'])
        closest_school_dists.to_pickle(PATH_TO_CACHED_SCHOOL_DISTS)

    return dd.merge(df, closest_school_dists, on='Street Name', how='left')
