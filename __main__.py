import argparse
import os
from enum import Enum

import dask.dataframe as dd


class Tasks(Enum):
    TASK_1 = 'task-1'
    TASK_2 = 'task-2'


def main(task: str, dataset_path: str):
    if task == Tasks.TASK_1.value:
        df = dd.read_csv(dataset_path,
                         dtype={
                             'House Number': str,
                             'Intersecting Street': str,
                             'Issuer Command': str,
                             'Violation Description': str,
                             'Violation Legal Code': str
                         },
                         assume_missing=True)
        print(df.head())
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ny-parking-violations-analysis')

    # select parser for task
    subparsers = parser.add_subparsers(required=True, dest='task', help='Task to run')
    task1_parser = subparsers.add_parser(Tasks.TASK_1.value)
    task1_parser.add_argument('--dataset-path', type=str,
                              default=os.path.join(os.path.dirname(__file__), 'data/Parking_Violations_Issued_-_Fiscal_Year_2022.csv'),
                              help='Path to dataset')
    args = parser.parse_args()
    main(args.task, dataset_path=args.dataset_path)
