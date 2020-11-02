Model Training
==============

All functions related to training our models.

Script
******
You can call `train.py` directly, to perform training on a given data file.

You can provide:

* The kind of model to use.
  This must be the first argument.
  If unspecified, defaults to `xgboost`.
  Can be: "xgboost", "random_forest" or "gradient_boosting".

* The path of the model, relative to the `data` folder.
  This must be the second argument.
  If unspecified, defaults to `train_median.csv`.

* Arguments to provide to the model.
  They must be after the two first arguments.
  They take the following format: `argument=value`.

The model will be saved with MlFlow.

Example:

.. code-block:: python

  ## Train on train_mean.csv, using a random forest with 40 estimators
  # On Windows
  python train.py random_forest train_mean.csv n_estimators=40

  # On Unix-based OS
  python3 train.py random_forest train_mean.csv n_estimators=40

If you want to see the results interactively, use `mlflow ui`.

Methods
*******

.. automodule:: src.train
   :members:
