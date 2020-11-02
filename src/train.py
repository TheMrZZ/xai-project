import sys
from typing import Tuple
from urllib.parse import urlparse

import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

try:
    from utils import get_data_path
except ImportError:
    from .utils import get_data_path


def eval_metrics(y_true, y_pred) -> Tuple[float, float, int, int, int, int]:
    """
    Returns 3 kind of evaluation metrics, based on a given model prediction:
    - The accuracy
    - The F1 score
    - The confusion matrix values

    :param y_true: The real values.
    :param y_pred: The values predicted by the model.
    :return: Accuracy, F1 score, True positives, False positives, True negatives, False negatives in that order.
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return accuracy, f1, tp, fp, tn, fn


def train_model(model: str, *, path: str = None, data: pd.DataFrame = None, model_kwargs={}):
    """
    Train a given type of model on given data.

    You must either specify a path or a dataframe.

    :param model: The kind of model to train. "xgboost", "random_forest" or "gradient_boosting".
    :param path: The path to the .csv file, to train the model on.
    :param data: The data to train the model on. They will be splitted with a 25% ratio.
    :param model_kwargs: The arguments to provide to the model.
    :return:
    """
    model = model.lower().strip()
    np.random.seed(40)

    for k, v in model_kwargs.items():
        # the model_kwargs are considered as strings but some are ints or floats, so we try to cast them as int or float
        try:
            model_kwargs[k] = int(v)
        except ValueError:
            try:
                model_kwargs[k] = float(v)
            except ValueError:
                pass  # Keep the string
    # for k, v in model_kwargs.items():
    #    print(type(k),type(v))

    if data is not None and path is not None:
        raise ValueError('You must specify either a path or a dataframe, not both.')

    if data is None:
        if path is None:
            raise ValueError('You must either specify a path or a dataframe.')

        data = pd.read_csv(path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    X = data.drop('TARGET', axis=1)
    y = data['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    with mlflow.start_run(run_name=model):
        # try:
        if model == 'xgboost':
            model = XGBClassifier(**model_kwargs)
        elif model == 'random_forest':
            model = RandomForestClassifier(**model_kwargs)
        elif model == 'gradient_boosting':
            model = GradientBoostingClassifier(**model_kwargs)
        else:
            raise ValueError(
                f'Unknown model "{model}", expected one of "xgboost", "random_forest" or "gradient_boosting"'
            )

        print('Training model...')
        model.fit(X_train, y_train)

        print('Calculating metrics...')
        y_pred = model.predict(X_test)

        accuracy, f1, tp, fp, tn, fn = eval_metrics(y_test, y_pred)

        print(f"  Accuracy:       {accuracy}")
        print(f"  F1 Score:       {f1}")
        print(f"  True positive:  {tp}")
        print(f"  False positive: {fp}")
        print(f"  True negative:  {tn}")
        print(f"  False negative: {fn}")

        mlflow.log_param("model", model)
        for param, value in model_kwargs.items():
            mlflow.log_param(param, str(value))

        if path is not None:
            mlflow.log_param("path", path)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("true_positive", tp)
        mlflow.log_metric("false_positive", fp)
        mlflow.log_metric("true_negative", tn)
        mlflow.log_metric("false_negative", fn)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        print('Registering model...')
        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model, "model", registered_model_name=model)
        else:
            mlflow.sklearn.log_model(model, "model")

    print('Over.')
    return model


def _main():
    model = sys.argv[1] if len(sys.argv) > 1 else 'xgboost'

    print('Loading data...')
    csv = get_data_path(sys.argv[2] if len(sys.argv) > 2 else 'train_median.csv')

    train_model(model, path=csv, model_kwargs=dict(arg.split('=') for arg in sys.argv[3:]))


if __name__ == "__main__":
    _main()
