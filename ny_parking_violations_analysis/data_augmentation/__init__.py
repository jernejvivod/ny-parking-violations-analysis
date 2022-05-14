import os
from enum import Enum


class DataAugEnum(Enum):
    WEATHER = 'weather'
    SCHOOLS = 'schools'
    EVENTS = 'events'
    BUSINESSES = 'businesses'
    ATTRACTIONS = 'attractions'


PATH_TO_CACHED_STREET_COORDINATES = os.path.join(os.path.dirname(__file__), 'cached/street_coordinates_dict.pkl')
PATH_TO_CACHED_UNIQUE_STREETS = os.path.join(os.path.dirname(__file__), 'cached/unique_streets.pkl')
PATH_TO_CACHED_ATTRACTIONS = os.path.join(os.path.dirname(__file__), 'cached/attractions_df.pkl')
PATH_TO_CACHED_ATTRACTIONS_DISTS = os.path.join(os.path.dirname(__file__), 'cached/attractions_dists_df.pkl')
PATH_TO_CACHED_SCHOOL_DISTS = os.path.join(os.path.dirname(__file__), 'cached/school_dists_df.pkl')
PATH_TO_CACHED_BUSINESSES = os.path.join(os.path.dirname(__file__), 'cached/businesses_df.pkl')
PATH_TO_CACHED_BUSINESSES_DISTS = os.path.join(os.path.dirname(__file__), 'cached/businesses_dists_df.pkl')
PATH_TO_CACHED_EVENTS = os.path.join(os.path.dirname(__file__), 'cached/events_df.pkl')
PATH_TO_CACHED_CONCERTS = os.path.join(os.path.dirname(__file__), 'cached/concerts_df.pkl')
PATH_TO_CACHED_GEOCODED_CONCERTS = os.path.join(os.path.dirname(__file__), 'cached/concerts_geocoded_df.pkl')

PATH_TO_AUGMENTED_DATASET_CSV = os.path.join(os.path.dirname(__file__), '../../data/dataset_augmented.csv')
