import argparse
import glob
import logging
import os
from contextlib import suppress
from enum import IntEnum

import dask_jobqueue
import dask_memusage
import matplotlib.pyplot as plt
from distributed import Client, LocalCluster

from ny_parking_violations_analysis import DATASET_AVRO_PATH, SCHEMA_FOR_AVRO, DATASET_PARQUET_PATH, DATASET_HDF_PATH, DATASET_HDF_KEY, BASE_DATASET_DEFAULT_PATH, read_parquet, is_county_code_valid
from ny_parking_violations_analysis import OutputFormat
from ny_parking_violations_analysis import Tasks, MLTask
from ny_parking_violations_analysis import read_base_dataset
from ny_parking_violations_analysis.data_augmentation import DataAugEnum, PATH_TO_AUGMENTED_DATASET_PARQUET, PATH_TO_AUGMENTED_DATASET_CSV
from ny_parking_violations_analysis.data_augmentation.augment import get_augmented_dataset
from ny_parking_violations_analysis.exploratory_analysis.analysis import groupby_count, plot_bar
from ny_parking_violations_analysis.exploratory_analysis.utilities import map_code_to_description
from ny_parking_violations_analysis.ml.tasks.ml import evaluate_violations_for_day, evaluate_car_make
from ny_parking_violations_analysis.streaming_analysis.streaming import stream, stream_clustering

# logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(**kwargs):
    """The main function that orchestrates the implemented functionality based on the command-line parameters.

    See the readme in the project root folder for instructions on how to run.
    """

    if not os.path.isdir(kwargs['mem_usage_path']):
        raise ValueError('{0} is not a directory'.format(kwargs['mem_usage_path']))

    if kwargs['use_slurm_cluster']:
        cluster = dask_jobqueue.SLURMCluster(
            queue=kwargs['queue'],
            n_workers=kwargs['n_workers'],
            processes=kwargs['processes'],
            cores=kwargs['cores'],
            memory=kwargs['memory'],
            scheduler_options={'dashboard_address': ':8087'},
            death_timeout=kwargs['death_timeout']
        )
        # Problems!
        # dask_memusage.install(cluster.scheduler, os.path.join(kwargs['mem_usage_path'], 'mem_usage.csv'))
        client = Client(cluster, timeout="240s")
        if not kwargs['no_scale']:
            client.cluster.scale(n=kwargs['n_workers'], jobs=kwargs['n_jobs'], cores=kwargs['cores'], memory=kwargs['memory'])
    else:
        cluster = LocalCluster()
        client = Client()
        dask_memusage.install(cluster.scheduler, os.path.join(kwargs['mem_usage_path'], 'mem_usage.csv'))

    with suppress(Exception):
        client.shutdown()

    # TASK 1
    if kwargs['task'] == Tasks.TASK_1.value:

        logger.info('Running task 1 (data format comparison)')

        # read base dataset and save in different format (if the files do not yet exist)
        df = read_base_dataset(kwargs['dataset_path'], parse_date=False)
        if not glob.glob(DATASET_AVRO_PATH):
            df.to_bag(format='dict').to_avro(DATASET_AVRO_PATH, SCHEMA_FOR_AVRO, compute=True)
        if not glob.glob(DATASET_PARQUET_PATH + ('/' if DATASET_PARQUET_PATH[-1] != '/' else '') + 'part.*.parquet'):
            df.to_parquet(DATASET_PARQUET_PATH, compute=True)
        if not glob.glob(DATASET_HDF_PATH):
            df.to_hdf(DATASET_HDF_PATH, DATASET_HDF_KEY, compute=True)

        # plot sizes
        sizes = dict()
        _MB_BYTES = 1048576
        sizes['csv'] = os.stat(kwargs['dataset_path']).st_size / _MB_BYTES
        sizes['Avro'] = sum(map(lambda x: os.stat(x).st_size / _MB_BYTES, glob.glob(DATASET_AVRO_PATH)))
        sizes['Parquet'] = sum(map(lambda x: os.stat(x).st_size / _MB_BYTES, glob.glob(DATASET_PARQUET_PATH)))
        sizes['HDF'] = sum(map(lambda x: os.stat(x).st_size / _MB_BYTES, glob.glob(DATASET_HDF_PATH)))
        plt.bar(*zip(*sizes.items()))
        plt.ylabel('MB')
        if os.path.isdir(kwargs['plot_dir_path']):
            plt.savefig(os.path.join(kwargs['plot_dir_path'], 'format_size_comparison.svg'))
        else:
            raise ValueError('\'{0}\' is not a directory'.format(kwargs['plot_dir_path']))

    # TASK 2
    elif kwargs['task'] == Tasks.TASK_2.value:

        logger.info('Running task 2 (base dataset augmentation)')

        # compute and save augmented dataset
        augmented_dataset = get_augmented_dataset(kwargs['dataset_path'], data_augmentations=kwargs['augmentations'])
        if kwargs['output_format'] == OutputFormat.PARQUET.value:
            logger.info('Saving augmented dataset in Parquet format to {0}'.format(PATH_TO_AUGMENTED_DATASET_PARQUET))
            augmented_dataset.to_parquet(PATH_TO_AUGMENTED_DATASET_PARQUET)
        elif kwargs['output_format'] == OutputFormat.CSV.value:
            logger.info('Saving augmented dataset in CSV format to {0}'.format(PATH_TO_AUGMENTED_DATASET_CSV))
            augmented_dataset.to_csv(PATH_TO_AUGMENTED_DATASET_CSV, single_file=True)
        else:
            raise NotImplementedError('output format \'{0}\' not recognized')

    # TASK 3
    elif kwargs['task'] == Tasks.TASK_3.value:

        logger.info('Running task 2 (exploratory data analysis)')

        # parse dataset
        df = read_base_dataset(kwargs['dataset_path'])

        # violations per day of week
        df['Weekday'] = df['Issue Date'].dt.weekday
        violations_per_day = groupby_count(df, 'Weekday', 7)
        violations_per_day = violations_per_day.sort_values('Weekday')
        plot_bar(
            violations_per_day, 'Weekday', 'Ticket Count', 'violations_per_weekday.png'
        )

        # violations per day in year
        violations_per_day = groupby_count(df, 'Issue Date', 365)
        violations_per_day = violations_per_day.sort_values('Issue Date')
        plot_bar(
            violations_per_day, 'Issue Date', 'Ticket Count', 'violations_per_day.png'
        )

        # top 10 violations types
        violations_per_type = groupby_count(df, 'Violation Code')
        violations_per_type['Violation Description'] = list(
            map(
                lambda x: map_code_to_description(x),
                violations_per_type['Violation Code'],
            )
        )
        plot_bar(
            violations_per_type,
            'Violation Description',
            'Ticket Count',
            'top_violation_types.png',
        )

        # top 10 counties with most violations
        violations_per_county = groupby_count(df, 'Violation County')
        plot_bar(
            violations_per_county,
            'Violation County',
            'Ticket Count',
            'top_violation_counties.png',
        )

        # top 10 streets with most violations
        violations_per_street = groupby_count(df, 'Street Name')
        plot_bar(
            violations_per_street,
            'Street Name',
            'Ticket Count',
            'top_violations_per_street.png',
        )

        # top 10 states with most violations outside NY
        violations_per_state = groupby_count(df, 'Registration State', 11)[1:]
        plot_bar(
            violations_per_state,
            'Registration State',
            'Ticket Count',
            'top_violation_states_outside_NY.png',
        )

        # top 10 peope/plates with most violations
        violations_per_plate = groupby_count(df, 'Plate ID')
        plot_bar(
            violations_per_plate,
            'Plate ID',
            'Ticket Count',
            'top_violations_per_plate.png',
        )

        # top 10 peope/plates with most violations without blank plate
        violations_per_plate = groupby_count(df, 'Plate ID', 11)[1:]
        plot_bar(
            violations_per_plate,
            'Plate ID',
            'Ticket Count',
            'top_violations_per_plate_no_blanks.png',
        )

    # TASK 4
    elif kwargs['task'] == Tasks.TASK_4.value:

        class Columns(IntEnum):
            DATE = 4
            BOROUGH = 21
            STREET = 24

        date_stats = stream(Columns.DATE, kwargs['dataset_path'])
        boroughs_stats = stream(Columns.BOROUGH, kwargs['dataset_path'])
        street_stats = stream(Columns.STREET, kwargs['dataset_path'])
        birch = stream_clustering([4, 21, 24], kwargs['dataset_path'])

        with open('task4_results.txt', 'a') as f:
            f.write(f'DATE\nMEAN: {date_stats.loc[0, "mean"]}, STD: {date_stats.loc[0, "std"]} \n')
            f.write(f'BOROUGHS\nMEAN: {boroughs_stats.loc[0, "mean"]}, STD: {boroughs_stats.loc[0, "std"]} \n')
            f.write(f'STREETS\nMEAN: {street_stats.loc[0, "mean"]}, STD: {street_stats.loc[0, "std"]} \n')
            print(f'CENTROIDS WITH BIRCH CLUSTERING: {birch["cluster"]} \n')

        logger.info('Running task 4 (stream-based data analysis)')

    # TASK 5
    elif kwargs['task'] == Tasks.TASK_5.value:

        logger.info('Running task 5 (machine learning)')

        # parse (augmented) dataset
        dataset = read_parquet(kwargs['dataset_path'])

        if not os.path.isdir(kwargs['res_path']):
            raise ValueError('{0} is not a directory'.format(kwargs['res_path']))

        # filter by county if applicable
        if kwargs['county_filter'] != 'ALL':
            county = kwargs['county_filter']
            if is_county_code_valid(county):
                dataset = dataset.loc[dataset['Violation County'] == county, :]
            else:
                raise ValueError('county with code {0} not recognized'.format(county))

        # perform evaluations for the ML tasks
        if kwargs['ml_task'] == MLTask.VIOLATIONS_FOR_DAY.value:
            if kwargs['reg_or_clf'] == 'reg':
                evaluate_violations_for_day(client, dataset, reg_or_clf='reg', alg=kwargs['alg'], res_path=kwargs['res_path'])
            elif kwargs['reg_or_clf'] == 'clf':
                evaluate_violations_for_day(client, dataset, reg_or_clf='clf', alg=kwargs['alg'], res_path=kwargs['res_path'])
            else:
                raise NotImplementedError('only \'reg\' or \'clf\' options are supported.')

        elif kwargs['ml_task'] == MLTask.CAR_MAKE.value:
            evaluate_car_make(client, dataset, car_make_filter=kwargs['car_make_filter'], alg=kwargs['alg'], res_path='.')
        else:
            raise NotImplementedError('option {0} not recognized. Only options {1} are supported.'.format(kwargs['ml_task'], ', '.join([v.value for v in MLTask])))

    elif kwargs['task'] == Tasks.SANITY_CHECK.value:
        logger.info('Running sanity check (compute a simple group-by task)')

        # run sanity check computation
        df = read_base_dataset(kwargs['dataset_path'], parse_date=True)
        logger.info('Dataframe has {0} partition(s).'.format(df.npartitions))
        df.groupby('Vehicle Make').size().compute()

        logger.info('Computation finished')

    else:
        raise NotImplementedError('option {0} not recognized. Only options {1} are supported.'.format(kwargs['task'], ','.join([v.value for v in MLTask])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ny-parking-violations-analysis')

    parser.add_argument('--use-slurm-cluster', action='store_true', help='Use SLURM cluster')
    parser.add_argument('--queue', type=str, default='all', help='Destination queue for each worker job. Passed to #SBATCH -p option.')
    parser.add_argument('--n_workers', type=int, default=8, help='Number of workers to start by default')
    parser.add_argument('--processes', type=int, default=1, help='Cut the job up into this many processes')
    parser.add_argument('--cores', type=int, default=8, help='Total number of cores per job')
    parser.add_argument('--memory', default='64GB', help='Total amount of memory per job')
    parser.add_argument('--death_timeout', type=int, default=120, help='Seconds to wait for a scheduler before closing workers')
    parser.add_argument('--n-jobs', type=int, default=3, help='For how many jobs to ask')
    parser.add_argument('--no-scale', action='store_true', help='Do not scale cluster at the start')
    parser.add_argument('--mem-usage-path', type=str, default='.', help='Path to folder in which to save memory usage data')

    # select parser for task
    subparsers = parser.add_subparsers(required=True, dest='task', help='Task to run')

    # TASK 1
    task1_parser = subparsers.add_parser(Tasks.TASK_1.value)

    task1_parser.add_argument('--dataset-path', type=str,
                              default=os.path.join(os.path.dirname(__file__), BASE_DATASET_DEFAULT_PATH),
                              help='Path to dataset')

    task1_parser.add_argument("--plot-dir-path", type=str, default='.', help='Path to folder in which to save plots')

    # TASK 2
    task2_parser = subparsers.add_parser(Tasks.TASK_2.value)

    task2_parser.add_argument('--dataset-path', type=str,
                              default=os.path.join(os.path.dirname(__file__), BASE_DATASET_DEFAULT_PATH),
                              help='Path to dataset')

    task2_parser.add_argument('--augmentations', type=str, nargs='+',
                              default=[e.value for e in DataAugEnum],
                              choices=[e.value for e in DataAugEnum],
                              help='additional data to add to the original dataset')

    task2_parser.add_argument('--output-format', type=str,
                              default=OutputFormat.PARQUET.value,
                              choices=[e.value for e in OutputFormat],
                              help='Path to dataset')

    # TASK 3
    task3_parser = subparsers.add_parser(Tasks.TASK_3.value)

    task3_parser.add_argument('--dataset-path', type=str,
                              default=os.path.join(os.path.dirname(__file__), PATH_TO_AUGMENTED_DATASET_PARQUET),
                              help='Path to dataset in Parquet format')

    # TASK 4
    task4_parser = subparsers.add_parser(Tasks.TASK_4.value)

    task4_parser.add_argument('--dataset-path', type=str,
                              default=os.path.join(os.path.dirname(__file__), BASE_DATASET_DEFAULT_PATH),
                              help='Path to dataset')

    # TASK 5
    task5_parser = subparsers.add_parser(Tasks.TASK_5.value)

    task5_parser.add_argument('--dataset-path', type=str,
                              default=os.path.join(os.path.dirname(__file__), PATH_TO_AUGMENTED_DATASET_PARQUET),
                              help='Path to dataset in Parquet format')

    task5_parser.add_argument('--county-filter', type=str, default='ALL',
                              help='Violations with which county code to consider. Specify \'ALL\' to include all including those with missing county code.')

    task5_parser.add_argument('--ml-task', type=str, default=MLTask.VIOLATIONS_FOR_DAY.value, help='ML task to run')

    task5_parser.add_argument('--reg-or-clf', type=str, default='clf', help='task to solve when evaluating the violations per day task (\'reg\' for the first task, \'clf\' for the second task')

    task5_parser.add_argument('--car-make-filter', nargs='+', type=str, default=['BMW', 'ME/BE'], help='task to solve when evaluating the violations per day task (\'reg\' for the first task, \'clf\' for the second task')

    task5_parser.add_argument('--res-path', type=str, default='.', help='path to folder for storing obtained results (plots and text files)')

    task5_parser.add_argument('--alg', type=str, default='xgb', help='algorithm to use (specified in the task instructions - \'partial_fit\', \'dask_ml\', \'xgb\' or \'dummy\')')

    sanity_check_parser = subparsers.add_parser(Tasks.SANITY_CHECK.value)

    sanity_check_parser.add_argument('--dataset-path', type=str,
                                     default=os.path.join(os.path.dirname(__file__), BASE_DATASET_DEFAULT_PATH),
                                     help='Path to dataset')

    args = parser.parse_args()

    main(**vars(args))
