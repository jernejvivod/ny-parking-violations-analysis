import os

import dask.dataframe as dd


def read_base_dataset(base_dataset_path: str):
    return dd.read_csv(base_dataset_path,
                       dtype={
                           'House Number': str,
                           'Intersecting Street': str,
                           'Issuer Command': str,
                           'Issuer Squad': str,
                           'Violation Post Code': str,
                           'Violation Description': str,
                           'Violation Legal Code': str
                       },
                       assume_missing=True)


def get_env_data_as_dict(path: str) -> dict:
    with open(path, 'r') as f:
        return dict(tuple(line.replace('\n', '').split('=')) for line in f.readlines() if not line.startswith('#'))


def get_google_api_key():
    return get_env_data_as_dict(os.path.join(os.path.dirname(__file__), '../.env'))['GOOGLE_API_KEY']
