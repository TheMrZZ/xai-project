Data Preparation
================

All functions related to cleaning, transforming and preparing the data.

Script
******
You can call `preparation.py` by using the following command:

.. code-block:: python

  # On Windows
  python preparation.py

  # On Unix-based OS
  python3 preparation.py

This will:

#. Load train & test data
#. Perform 2 kind of data preparation on them: mean & median.
#. Save the results in the data folder.

Methods
*******

.. automodule:: src.preparation
   :members:
