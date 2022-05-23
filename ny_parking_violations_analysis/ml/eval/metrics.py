from sklearn import metrics


def get_regression_metrics(y_test, y_pred, res_file_dir_path, res_file_name):
    r2_score = metrics.r2_score(y_test, y_pred)
    rmse_score = metrics.mean_squared_error(y_test, y_pred, squared=False)
    with open(res_file_dir_path + '/' + res_file_name + '.txt', 'w') as f:
        f.write('r2-score: {0}\nrmse-score: {1}\n'.format(round(r2_score, 3), round(rmse_score, 3)))
