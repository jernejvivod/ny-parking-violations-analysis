import os

from . import logger


def write_time(time: float, dir_path: str, file_name: str):
    output_file_path = os.path.abspath(os.path.join(dir_path, file_name + '.txt'))
    logger.info('Writing elapsed time to {0}'.format(output_file_path))

    with open(output_file_path, 'w') as f:
        f.write('elapsed training time: {0}\n'.format(round(time, 3)))
