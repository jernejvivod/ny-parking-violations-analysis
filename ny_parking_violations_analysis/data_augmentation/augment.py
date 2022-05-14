import os
import pickle as pkl
from typing import Iterable

import dask.dataframe as dd
import tqdm

import ny_parking_violations_analysis.data_augmentation.specific.attractions as attractions
import ny_parking_violations_analysis.data_augmentation.specific.businesses as businesses
import ny_parking_violations_analysis.data_augmentation.specific.events as events
import ny_parking_violations_analysis.data_augmentation.specific.schools as schools
import ny_parking_violations_analysis.data_augmentation.specific.weather as weather
from ny_parking_violations_analysis import get_google_api_key
from ny_parking_violations_analysis import read_base_dataset
from ny_parking_violations_analysis.data_augmentation import DataAugEnum, PATH_TO_CACHED_UNIQUE_STREETS, PATH_TO_CACHED_STREET_COORDINATES
from ny_parking_violations_analysis.data_augmentation.specific import get_unique_streets, name_to_coordinates


def get_augmented_dataset(base_dataset_path: str, data_augmentations: Iterable = tuple(e.value for e in DataAugEnum)) -> dd:
    """Get augmented dataset. Parse original dataset as a Dask Dataframe and add specified augmentations.

    :param base_dataset_path: path to original dataset
    :param data_augmentations: list of names of augmentations to add
    :return: Dask DataFrame representing the augmented dataset.
    """

    # parse base dataset
    df = read_base_dataset(base_dataset_path)
    df['Issue Date'] = dd.to_datetime(df['Issue Date'])

    # if street to coordinates mapping needed
    street_coordinates = None
    if any([aug in data_augmentations for aug in (DataAugEnum.SCHOOLS.value,
                                                  DataAugEnum.EVENTS.value,
                                                  DataAugEnum.BUSINESSES.value,
                                                  DataAugEnum.ATTRACTIONS.value
                                                  )]):

        # parse cached mapping if exists, else compute new and cache
        if os.path.exists(PATH_TO_CACHED_STREET_COORDINATES):
            with open(PATH_TO_CACHED_STREET_COORDINATES, 'rb') as f:
                street_coordinates = pkl.load(f)
        else:
            # parse cached list of unique streets if it exists, else compute new and cache
            if os.path.exists(PATH_TO_CACHED_UNIQUE_STREETS):
                with open(PATH_TO_CACHED_UNIQUE_STREETS, 'rb') as f:
                    unique_streets = pkl.load(f)
            else:
                unique_streets = get_unique_streets(df)
                with open(PATH_TO_CACHED_UNIQUE_STREETS, 'wb') as f:
                    pkl.dump(unique_streets, f)

            # map streets to coordinates
            street_coordinates = dict()
            google_api_key = get_google_api_key()
            for street in tqdm.tqdm(unique_streets, desc='Performing geocoding using the Google Maps API', unit='address'):
                if isinstance(street, str):
                    street_coordinates[street] = name_to_coordinates(street, 'NY', google_api_key)

            with open(PATH_TO_CACHED_STREET_COORDINATES, 'wb') as f:
                pkl.dump(street_coordinates, f)

    return add_augmentations(df, data_augmentations, street_coordinates)


def add_augmentations(df: dd, data_augmentations: Iterable, street_coordinates: dict = None) -> dd:
    """ Add specified augmentations to the main dataset.

    :param df: DataFrame representing the original dataset
    :param data_augmentations: iterable specifying the dataset augmentations
    :param street_coordinates: DataFrame containing the coordinates of unique streets in the original dataset
    :return: augmented dataset
    """

    # data augmentation implementations
    if DataAugEnum.WEATHER.value in data_augmentations:
        # WEATHER
        df = weather.join_with(df)
    if DataAugEnum.SCHOOLS.value in data_augmentations:
        # SCHOOLS
        df = schools.join_with(df, street_coordinates)
    if DataAugEnum.EVENTS.value in data_augmentations:
        # EVENTS
        df = events.join_with(df)
    if DataAugEnum.BUSINESSES.value in data_augmentations:
        # BUSINESSES
        df = businesses.join_with(df, street_coordinates)
    if DataAugEnum.ATTRACTIONS.value in data_augmentations:
        # ATTRACTIONS
        df = attractions.join_with(df, street_coordinates)
    return df
