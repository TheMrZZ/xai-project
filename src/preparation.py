from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

try:
    from utils import get_data_path
except ImportError:
    from .utils import get_data_path


def dummy_encode(df: pd.DataFrame, encoder: OneHotEncoder = None) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """
    One hot encode all string columns.

    :param df: The dataframe to one hot encode.
    :return: A new one hot encoded dataframe, and the corresponding encoder.
    """
    need_encoding = df.select_dtypes('object').fillna('None')

    if encoder is None:
        try:
            encoder = OneHotEncoder()
            encoder.fit(need_encoding)
        except Exception as e:
            print(need_encoding)
            raise e

    encoded_values: np.array = encoder.transform(need_encoding).toarray()
    features_names: Tuple[str] = encoder.get_feature_names(need_encoding.columns)

    restored_df = pd.concat([
        df.select_dtypes(exclude='object'),
        pd.DataFrame(encoded_values, columns=features_names).astype(int)
    ], axis=1)

    return restored_df, encoder


def clean_df(df: pd.DataFrame, kind: str = 'median') -> pd.DataFrame:
    """
    Clean the dataframe, by removing NaN using one of three strategies:
    - drop rows with at least one NaN
    - replace by the median of the column
    - replace by the mean of the column.

    :param df: The DataFrame
    :param kind: "drop", "median", or "mean".

    :return: A cleaned dataframe.
    """

    if kind == 'drop':
        return df.dropna()
    if kind == 'mean':
        return df.apply(lambda column: column.fillna(column.mean()))
    if kind == 'median':
        return df.apply(lambda column: column.fillna(column.median()))

    raise ValueError(f'Unknown kind "{kind}".')


def _main():
    print('Reading data...')
    train_df = pd.read_csv(get_data_path('application_train.csv'))
    test_df = pd.read_csv(get_data_path('application_test.csv'))

    print('Performing data cleaning & features engineering...')
    train_df_encoded, encoder = dummy_encode(train_df)
    test_df_encoded, _ = dummy_encode(test_df, encoder)

    # Replace NaN by mean
    train_df_mean = clean_df(train_df_encoded, 'mean')
    test_df_mean = clean_df(test_df_encoded, 'mean')

    # Replace NaN by median
    train_df_median = clean_df(train_df_encoded, 'median')
    test_df_median = clean_df(test_df_encoded, 'median')

    print('Saving data... (It can take a bit of time)')

    def save(data, path):
        print('  Saving', path)
        data.to_csv(get_data_path(path))
        print('  Data saved.')

    # Save all dataframes
    save(train_df_mean, 'train_mean.csv')
    save(test_df_mean, 'test_mean.csv')

    save(train_df_median, 'train_median.csv')
    save(test_df_median, 'test_median.csv')

    print('Over.')


if __name__ == '__main__':
    _main()
