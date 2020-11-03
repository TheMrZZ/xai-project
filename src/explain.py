import sys

import mlflow
import pandas as pd
import shap
from xgboost.sklearn import XGBClassifier

try:
    from utils import get_data_path
except ImportError:
    from .utils import get_data_path


def explain(model: XGBClassifier, df: pd.DataFrame, frac: float = 0.1) -> None:
    """
    Explains a given model.

    :param model: The model to explain.
    :param df: The dataframe to use to explain.
    :param frac: The percentage of the data to use when explaining "all" points.
    Can be set to a low number if it takes too much time to run.
    """
    explainer = shap.TreeExplainer(model)

    print('Explaining values...')
    samples = df.sample(1000)
    shap_values = explainer.shap_values(samples)

    print('Plotting...')
    try:
        iter(explainer.expected_value)
        expected_values = explainer.expected_value[0]
        shap_values = shap_values[0]
    except TypeError:
        expected_values = explainer.expected_value
        shap_values = shap_values

    # Explain for a single point
    shap.save_html(
        'single_value_plot.html',
        shap.force_plot(expected_values, shap_values[2], samples.iloc[2])
    )
    # Explain for all points
    shap.save_html('multiple_values_plot.html', shap.force_plot(explainer.expected_value, shap_values, samples))
    # Summary plot
    shap.summary_plot(shap_values, samples)

    print('Plots saved to 3 html files.')


def _main(model_uri: str, df: pd.DataFrame):
    print('Loading model...')
    model = mlflow.sklearn.load_model(model_uri)
    print('Model loaded.')
    explain(model, df)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise ValueError('Missing model_id parameter. Example: python explain.py da7c69ed9ccf49bf8f509626f6f3e447')

    print('Loading data...')
    df_ = pd.read_csv(get_data_path('train_median.csv'))
    print('Data loaded.')

    model_id = sys.argv[1]
    _main(f'mlruns/0/{model_id}/artifacts/model', df_.drop('TARGET', axis=1))
