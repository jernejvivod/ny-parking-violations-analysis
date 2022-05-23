import os

from sklearn import metrics

from . import logger


def get_regression_metrics(y_test, y_pred, dir_path, file_name):
    output_file_path = os.path.abspath(os.path.join(dir_path, file_name + '.txt'))
    logger.info('Writing computed regression metrics to to {0}'.format(output_file_path))

    r2_score = metrics.r2_score(y_test, y_pred)
    rmse_score = metrics.mean_squared_error(y_test, y_pred, squared=False)
    with open(output_file_path, 'w') as f:
        f.write('r2-score: {0}\nrmse-score: {1}\n'.format(round(r2_score, 3), round(rmse_score, 3)))
