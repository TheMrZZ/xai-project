Explain
=======

Explain a given model.

It:

* Saves a first HTML file, where a specific point is explained.

* Saves a second HTML file, where all data points are explained.

* Plots a summary for each class of the whole dataset.

Script
******
You can call `explain.py` directly, to explain a given model.

You must provide the id of the mlflow run of the model to use.

Example:

.. code-block:: python

  ## Process the data, replacing NaN by means, using a random forest with 40 estimators
  # On Windows
  python explain.py da7c69ed9ccf49bf8f509626f6f3e447

  # On Unix-based OS
  python3 explain.py da7c69ed9ccf49bf8f509626f6f3e447

Methods
*******

.. automodule:: src.explain
   :members:
