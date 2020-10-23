# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

from urllib.parse import urlparse

import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier

from utils import get_data_path


def eval_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return accuracy, f1, tp, fp, tn, fn


classifiers = {
    'xgboost': XGBClassifier(),
    'random_forest': RandomForestClassifier(),
    'gradient_boosting': GradientBoostingClassifier(),
}


def main(model: str):
    model = model.lower().strip()
    np.random.seed(40)

    print('Reading data...')
    data = pd.read_csv(get_data_path('train_median.csv'))

    # Split the data into training and test sets. (0.75, 0.25) split.
    X = data.drop('TARGET', axis=1)
    y = data['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    with mlflow.start_run():
        try:
            model = classifiers[model]
        except KeyError:
            raise ValueError(f'Unknown model "{model}", expected one of {list(classifiers.keys())}')

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


if __name__ == "__main__":
    main('xgboost')
