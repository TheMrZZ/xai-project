import pandas as pd

from utils import get_data_path


def dummy_encode(df: pd.DataFrame) -> pd.DataFrame:
    """ Dummy encode all string columns. """
    columns_to_encode = [col for col in df.columns if df[col].dtype == 'object']
    return pd.get_dummies(
        df,
        prefix=columns_to_encode,
        columns=columns_to_encode
    )


def clean_df(df: pd.DataFrame, kind: str = 'median') -> pd.DataFrame:
    """
    Clean the dataframe & perform features engineering.

    :param df: The DataFrame
    :param kind: "drop", "median", or "mean".
    """

    if kind == 'drop':
        return df.dropna()
    if kind == 'mean':
        return df.apply(lambda column: column.fillna(column.mean()))
    if kind == 'median':
        return df.apply(lambda column: column.fillna(column.median()))

    raise ValueError(f'Unknown kind "{kind}".')


if __name__ == '__main__':
    print('Reading data...')
    train_df = pd.read_csv(get_data_path('application_train.csv'))
    test_df = pd.read_csv(get_data_path('application_test.csv'))

    print('Performing data cleaning & features engineering...')
    train_df_encoded = dummy_encode(train_df)
    test_df_encoded = dummy_encode(test_df)

    # Replace NaN by mean
    train_df_mean = clean_df(train_df_encoded, 'mean')
    test_df_mean = clean_df(test_df_encoded, 'mean')

    # Replace NaN by median
    train_df_median = clean_df(train_df_encoded, 'median')
    test_df_median = clean_df(test_df_encoded, 'median')

    print('Saving data...')
    # Save all dataframes
    train_df_mean.to_csv(get_data_path('train_mean.csv'))
    test_df_mean.to_csv(get_data_path('test_mean.csv'))

    train_df_median.to_csv(get_data_path('train_median.csv'))
    test_df_median.to_csv(get_data_path('test_median.csv'))

    print('Over.')
