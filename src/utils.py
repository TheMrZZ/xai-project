from os import path

try:
    import config
except ImportError:
    from . import config


def get_data_path(file_name: str) -> str:
    """
    Returns the path of a given data file, located in the DATA_FOLDER.

    :param file_name: The name of the data file.

    :return: The path of the data file.
    """
    return path.join(config.DATA_FOLDER, file_name)
