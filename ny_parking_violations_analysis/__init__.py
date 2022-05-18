import os
from enum import Enum

import dask.dataframe as dd

BASE_DATASET_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), '../data/Parking_Violations_Issued_-_Fiscal_Year_2022.csv')
DATASET_AVRO_PATH = os.path.join(os.path.dirname(__file__), '../data/avro/dataset.*.avro')
DATASET_PARQUET_PATH = os.path.join(os.path.dirname(__file__), '../data/parquet/')
DATASET_HDF_PATH = os.path.join(os.path.dirname(__file__), '../data/hdf/output-*.hdf')
DATASET_HDF_KEY = '/data'


class Tasks(Enum):
    TASK_1 = 'task-1'
    TASK_2 = 'task-2'
    TASK_3 = 'task-3'
    TASK_5 = 'task-5'


class MLTask(Enum):
    VIOLATIONS_FOR_DAY = 'tickets_for_day'
    CAR_MAKE = 'car_make'


class OutputFormat(Enum):
    PARQUET = 'parquet'
    CSV = 'csv'


SCHEMA_FOR_AVRO = {'name': 'Violations', 'doc': 'Parking Violations Dataset',
                   'type': 'record',
                   'fields': [
                       {'name': 'Summons Number', 'type': ['string', 'float']},
                       {'name': 'Plate ID', 'type': ['string', 'float']},
                       {'name': 'Registration State', 'type': ['string', 'float']},
                       {'name': 'Plate Type', 'type': ['string', 'float']},
                       {'name': 'Issue Date', 'type': ['string', 'float']},
                       {'name': 'Violation Code', 'type': ['string', 'float']},
                       {'name': 'Vehicle Body Type', 'type': ['string', 'float']},
                       {'name': 'Vehicle Make', 'type': ['string', 'float']},
                       {'name': 'Issuing Agency', 'type': ['string', 'float']},
                       {'name': 'Street Code1', 'type': ['string', 'float']},
                       {'name': 'Street Code2', 'type': ['string', 'float']},
                       {'name': 'Street Code3', 'type': ['string', 'float']},
                       {'name': 'Vehicle Expiration Date', 'type': ['string', 'float']},
                       {'name': 'Violation Location', 'type': ['string', 'float']},
                       {'name': 'Violation Precinct', 'type': ['string', 'float']},
                       {'name': 'Issuer Precinct', 'type': ['string', 'float']},
                       {'name': 'Issuer Code', 'type': ['string', 'float']},
                       {'name': 'Issuer Command', 'type': ['string', 'float']},
                       {'name': 'Issuer Squad', 'type': ['string', 'float']},
                       {'name': 'Violation Time', 'type': ['string', 'float']},
                       {'name': 'Time First Observed', 'type': ['string', 'float']},
                       {'name': 'Violation County', 'type': ['string', 'float']},
                       {'name': 'Violation In Front Of Or Opposite', 'type': ['string', 'float']},
                       {'name': 'House Number', 'type': ['string', 'float']},
                       {'name': 'Street Name', 'type': ['string', 'float']},
                       {'name': 'Intersecting Street', 'type': ['string', 'float']},
                       {'name': 'Date First Observed', 'type': 'float'},
                       {'name': 'Law Section', 'type': 'float'},
                       {'name': 'Sub Division', 'type': ['string', 'float']},
                       {'name': 'Violation Legal Code', 'type': ['string', 'float']},
                       {'name': 'Days Parking In Effect    ', 'type': ['string', 'float']},
                       {'name': 'From Hours In Effect', 'type': ['string', 'float']},
                       {'name': 'To Hours In Effect', 'type': ['string', 'float']},
                       {'name': 'Vehicle Color', 'type': ['string', 'float']},
                       {'name': 'Unregistered Vehicle?', 'type': ['string', 'float']},
                       {'name': 'Vehicle Year', 'type': 'float'},
                       {'name': 'Meter Number', 'type': ['string', 'float']},
                       {'name': 'Feet From Curb', 'type': 'float'},
                       {'name': 'Violation Post Code', 'type': ['string', 'float']},
                       {'name': 'Violation Description', 'type': ['string', 'float']},
                       {'name': 'No Standing or Stopping Violation', 'type': ['string', 'float']},
                       {'name': 'Hydrant Violation', 'type': ['string', 'float']},
                       {'name': 'Double Parking Violation', 'type': ['string', 'float']}
                   ]}


def read_base_dataset(base_dataset_path: str, parse_date=True) -> dd:
    df = dd.read_csv(base_dataset_path,
                     dtype={
                         'Summons Number': str,
                         'Plate ID': str,
                         'Registration State': str,
                         'Plate Type': str,
                         'Issue Date': str,
                         'Violation Code': str,
                         'Vehicle Body Type': str,
                         'Vehicle Make': str,
                         'Issuing Agency': str,
                         'Street Code1': str,
                         'Street Code2': str,
                         'Street Code3': str,
                         'Vehicle Expiration Date': str,
                         'Violation Location': str,
                         'Violation Precinct': str,
                         'Issuer Precinct': str,
                         'Issuer Code': str,
                         'Issuer Command': str,
                         'Issuer Squad': str,
                         'Violation Time': str,
                         'Time First Observed': str,
                         'Violation County': str,
                         'Violation In Front Of Or Opposite': str,
                         'House Number': str,
                         'Street Name': str,
                         'Intersecting Street': str,
                         'Date First Observed': float,
                         'Law Section': float,
                         'Sub Division': str,
                         'Violation Legal Code': str,
                         'Days Parking In Effect    ': str,
                         'From Hours In Effect': str,
                         'To Hours In Effect': str,
                         'Vehicle Color': str,
                         'Unregistered Vehicle?': str,
                         'Vehicle Year': float,
                         'Meter Number': str,
                         'Feet From Curb': float,
                         'Violation Post Code': str,
                         'Violation Description': str,
                         'No Standing or Stopping Violation': str,
                         'Hydrant Violation': str,
                         'Double Parking Violation': str,
                     },
                     blocksize='512MB',
                     assume_missing=True)
    if parse_date:
        df['Issue Date'] = dd.to_datetime(df['Issue Date'])
    return df


def get_base_dataset_columns():
    return read_base_dataset(BASE_DATASET_DEFAULT_PATH).columns


def get_env_data_as_dict(path: str) -> dict:
    with open(path, 'r') as f:
        return dict(tuple(line.replace('\n', '').split('=')) for line in f.readlines() if not line.startswith('#'))


def get_google_api_key():
    return get_env_data_as_dict(os.path.join(os.path.dirname(__file__), '../.env'))['GOOGLE_API_KEY']
