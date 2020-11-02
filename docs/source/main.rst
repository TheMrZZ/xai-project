Main
====

Runs all the parts (loading, processing, training, explanation, prediction) on the data.

This will save the predicted values (on the train set) to DATA_FOLDER/predicted_values.csv.

Script
******
You can call `main.py` directly, to perform all operations on the data, with a given model & a given normalization.

You can provide:

* The kind of model to use.
  This must be the first argument.
  If unspecified, defaults to `xgboost`.
  Can be: "xgboost", "random_forest" or "gradient_boosting".

* The normalization method.
  This must be the second argument.
  If unspecified, defaults to `median`.
  Can be: "median" or "mean".

* Arguments to provide to the model.
  They must be after the two first arguments.
  They take the following format: `argument=value`.

Example:

.. code-block:: python

  ## Process the data, replacing NaN by means, using a random forest with 40 estimators
  # On Windows
  python main.py random_forest mean n_estimators=40

  # On Unix-based OS
  python3 train.py random_forest mean n_estimators=40

If you want to see the results interactively, use `mlflow ui`.

Methods
*******

.. automodule:: src.main
   :members:
