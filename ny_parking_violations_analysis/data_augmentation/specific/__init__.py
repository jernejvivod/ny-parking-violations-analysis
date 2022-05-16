import re
import time

import dask.dataframe as dd
import numpy as np
import pandas as pd
import requests
from geopy import distance

WEBSITE_BUSINESSES_PATH = 'https://www.newyorkupstate.com/news/erry-2018/06/fd8d4b2ff06757/new_yorks_100_largest_companie.html'
WEBSITE_ATTRACTIONS_PATH = 'https://tourscanner.com/blog/best-tourist-attractions-in-new-york-city/'
WEBSITE_CONCERTS_PATH_TEMPLATE = 'https://www.concertarchives.org/locations/new-york-ny--3?page={0}#concert-table'


def get_unique_streets(df: dd) -> list:
    return list(df['Street Name'].unique().compute())


def prepare_address(address: str, state_suff: str):
    address_single_whitespace = re.compile(r"\s+").sub(" ", address).strip()
    return address_single_whitespace.replace(' ', '+').replace('&', '%26').replace('\'', '%27') + ',+' + state_suff


def get_name_coordinates(address: str, google_api_key, max_retries=10):
    req_url = 'https://maps.googleapis.com/maps/api/geocode/json?address={0}&key={1}'.format(address, google_api_key)
    retries = 10
    while True:
        try:
            res = requests.get(req_url).json()
            time.sleep(0.001)
            if res['results']:
                return res['results'][0]['geometry']['location']
            else:
                return {'lat': 0.0, 'lng': 0.0}
        except Exception:
            if retries > max_retries:
                return {'lat': 0.0, 'lng': 0.0}
            retries += 1
            time.sleep(1)


def name_to_coordinates(address: str, state_suff: str, google_api_key: str):
    return get_name_coordinates(prepare_address(address, state_suff), google_api_key)


def compute_dist(coor1, coor2):
    try:
        return distance.geodesic(coor1, coor2).kilometers
    except ValueError:
        return np.inf


def get_dist_closest(coordinates: dict, df: pd.DataFrame, name_col: str = 'NAME'):
    return df.apply(lambda x: (compute_dist((coordinates['lat'], coordinates['lng']), (x['LATITUDE'], x['LONGITUDE'])), x[name_col]), axis=1).min()
