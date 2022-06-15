import os

from . import logger


def write_classification_report(cr: str, dir_path: str, file_name: str):
    output_file_path = os.path.abspath(os.path.join(dir_path, file_name + '.txt'))
    logger.info('Writing classification report to {0}'.format(output_file_path))
    with open(output_file_path, 'w') as f:
        f.write(cr)
