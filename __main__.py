import argparse
import glob
import os

import matplotlib.pyplot as plt
from dask_ml.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

from ny_parking_violations_analysis import BASE_DATASET_DEFAULT_PATH, DATASET_AVRO_PATH, DATASET_PARQUET_PATH, DATASET_HDF_PATH, DATASET_HDF_KEY
from ny_parking_violations_analysis import SCHEMA_FOR_AVRO
from ny_parking_violations_analysis import Tasks, OutputFormat, MLTask
from ny_parking_violations_analysis import read_base_dataset, get_base_dataset_columns
from ny_parking_violations_analysis.data_augmentation import PATH_TO_AUGMENTED_DATASET_CSV, DataAugEnum
from ny_parking_violations_analysis.data_augmentation.augment import get_augmented_dataset
from ny_parking_violations_analysis.ml.ml_pipeline import train_with_partial_fit
from ny_parking_violations_analysis.ml.transform_dataset import transform_for_training_day


def main(**kwargs):
    if kwargs['task'] == Tasks.TASK_1.value:
        df = read_base_dataset(kwargs['dataset_path'], parse_date=False)
        if not glob.glob(DATASET_AVRO_PATH):
            df.to_bag(format='dict').to_avro(DATASET_AVRO_PATH, SCHEMA_FOR_AVRO, compute=True)
        if not glob.glob(DATASET_PARQUET_PATH + ('/' if DATASET_PARQUET_PATH[-1] != '/' else '') + 'part.*.parquet'):
            df.to_parquet(DATASET_PARQUET_PATH, compute=True)
        if not glob.glob(DATASET_HDF_PATH):
            df.to_hdf(DATASET_HDF_PATH, DATASET_HDF_KEY, compute=True)
        sizes = dict()
        sizes['csv'] = os.stat(kwargs['dataset_path']).st_size / (1024 * 1024)
        sizes['Avro'] = sum(map(lambda x: os.stat(x).st_size / (1024 * 1024), glob.glob(DATASET_AVRO_PATH)))
        sizes['Parquet'] = sum(map(lambda x: os.stat(x).st_size / (1024 * 1024), glob.glob(DATASET_PARQUET_PATH)))
        sizes['HDF'] = sum(map(lambda x: os.stat(x).st_size / (1024 * 1024), glob.glob(DATASET_HDF_PATH)))
        plt.bar(*zip(*sizes.items()))
        plt.ylabel('MB')
        if os.path.isdir(kwargs['plot_dir_path']):
            plt.savefig(os.path.join(kwargs['plot_dir_path'], 'format_size_comparison.svg'))
        else:
            raise ValueError('\'{0}\' is not a directory'.format(kwargs['plot_dir_path']))

    elif kwargs['task'] == Tasks.TASK_2.value:
        # compute and save augmented dataset
        augmented_dataset = get_augmented_dataset(kwargs['dataset_path'], data_augmentations=kwargs['augmentations'])
        if kwargs['output_format'] == OutputFormat.PARQUET.value:
            augmented_dataset.to_parquet(PATH_TO_AUGMENTED_DATASET_CSV)
        elif kwargs['output_format'] == OutputFormat.CSV.value:
            augmented_dataset.to_csv(PATH_TO_AUGMENTED_DATASET_CSV, single_file=True)
        else:
            raise NotImplementedError()
    elif kwargs['task'] == Tasks.TASK_5.value:
        # compute and save augmented dataset
        if kwargs['ml_task'] == MLTask.VIOLATIONS_FOR_DAY.value:
            df = read_base_dataset(kwargs['dataset_path'])
            columns_for_violation = get_base_dataset_columns()
            df_transformed = transform_for_training_day(df, columns_for_violation, 3).repartition(partition_size='128MB').persist()  # Computed dataset is small. Can be persisted in memory.
            x_train, x_test, y_train, y_test = train_test_split(df_transformed.loc[:, df_transformed.columns != 'month'], df_transformed['month'], random_state=0)
            clf_1 = train_with_partial_fit(x_train, y_train, clf=SGDClassifier(), all_classes=df_transformed['month'].unique().compute())

            # TODO Dask ml

            # TODO XGBoost

            # TODO Evaluate on test set, compute useful metrics (is a regression problem)

            # Also turn into classification problem - compute average and classify if number of parking violations will be above or below average

        elif kwargs['ml_task'] == MLTask.CAR_MAKE.value:
            pass
            # TODO process data and save to disk
            # Read data from disk and perform training using the three methods (train the three classifiers)

            # evaluate on test set, print confusion matrix

    else:
        raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ny-parking-violations-analysis')

    # select parser for task
    subparsers = parser.add_subparsers(required=True, dest='task', help='Task to run')

    # TASK 1
    task1_parser = subparsers.add_parser(Tasks.TASK_1.value)
    task1_parser.add_argument('--dataset-path', type=str,
                              default=os.path.join(os.path.dirname(__file__), BASE_DATASET_DEFAULT_PATH),
                              help='Path to dataset')
    task1_parser.add_argument("--plot-dir-path", type=str, default='.', help='path to folder in which to save plots')

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

    # TASK 5
    task5_parser = subparsers.add_parser(Tasks.TASK_5.value)

    task5_parser.add_argument('--dataset-path', type=str,
                              default=os.path.join(os.path.dirname(__file__), BASE_DATASET_DEFAULT_PATH),
                              help='Path to dataset')
    task5_parser.add_argument('--ml-task', type=str, default=MLTask.VIOLATIONS_FOR_DAY.value, help='ML task to run')

    args = parser.parse_args()
    main(**vars(args))
