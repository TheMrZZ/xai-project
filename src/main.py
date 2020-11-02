import sys

import pandas as pd

try:
    from preparation import clean_df, dummy_encode
    from train import train_model
    from utils import get_data_path
    from explain import explain
except ImportError:
    from .preparation import clean_df, dummy_encode
    from .train import train_model
    from .utils import get_data_path
    from .explain import explain


def full_process(model_type: str, normalization_method: str, model_kwargs: dict = {}):
    """
    Run all the pipeline (loading, preparation, training, explanation, prediction) with a given model & a given normalization.

    :param model_type: The kind of model to train. "xgboost", "random_forest" or "gradient_boosting".
    :param normalization_method: How to handle NaN value. "mean" or "median".
    :param model_kwargs: The arguments to provide to the model.
    """
    print(
        f'Running all processes with the {model_type} model, and the {normalization_method} normalization.',
        f'Additional model params are {model_kwargs}.' if model_kwargs else ''
    )
    # Preparation
    print('Preparation...')
    train_df = pd.read_csv(get_data_path('application_train.csv'))
    test_df = pd.read_csv(get_data_path('application_test.csv'))

    print('Performing data cleaning & features engineering...')

    train_df, encoder = dummy_encode(train_df)
    test_df, _ = dummy_encode(test_df, encoder)

    if normalization_method in ('median', 'mean'):
        train_df = clean_df(train_df, normalization_method)
        test_df = clean_df(test_df, normalization_method)
    else:
        raise ValueError(f'Expected normalization_method to be "mean" or "median", got "{normalization_method}"')

    # Training
    print('Training...')
    model = train_model(model_type, data=train_df, model_kwargs=model_kwargs)

    # Explaining
    print('Explaining...')
    explain(model, train_df.drop('TARGET', axis=1))

    # Predict
    print('Predicting...')
    predicted_df = pd.concat([
        test_df,
        pd.DataFrame(model.predict(test_df), columns=['predicted']),
    ])

    # Save the prediction
    pred_path = get_data_path('predicted_values.csv')
    print('Saving prediction to', pred_path)
    predicted_df.to_csv(pred_path)

    print('Over.')


def _main():
    model = sys.argv[1] if len(sys.argv) > 1 else 'xgboost'
    normalization_type = sys.argv[2] if len(sys.argv) > 2 else 'median'

    full_process(model, normalization_type, model_kwargs=dict(arg.split('=') for arg in sys.argv[3:]))


if __name__ == '__main__':
    _main()
