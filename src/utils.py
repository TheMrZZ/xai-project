from os import path

ROOT_FOLDER = '..'
DATA_FOLDER = path.join(ROOT_FOLDER, 'data')


def get_data_path(file: str) -> str:
    return path.join(DATA_FOLDER, file)
