.. _environment:

===========
Environment
===========

To get the most out of MyAutoML you will need to install and setup several components in your environment for MyAutoML
to work with.


MyAutoML Process
----------------

The processes that we aim to cover with MyAutoML are as follows:

.. figure:: ../images/training-process.png
    :width: 100%
    :align: center

and

.. figure:: ../images/prediction-process.png
    :width: 100%
    :align: center

There are two separate processes, one for training a model and one for making predictions. Each process is executed
by running a Python script, e.g. :code:`train.py` and :code:`predict.py`. This can be as simple or as complex as you
like: you can run the scripts manually (you can even run the code from a Jupyter notebook), or as an automated script
in a Docker container on a Kubernetes platform scheduled by Airflow. A prediction script can make predictions for a
batch of items, or it can spawn an API for real-time, on-demand predictions.


Software components needed
--------------------------

Model Registry: MLflow
~~~~~~~~~~~~~~~~~~~~~~

As a model registry we work with `MLflow <https://mlflow.org>`__. Please refer to the
`installation instructions <https://mlflow.org/docs/latest/quickstart.html#installing-mlflow>`__ and
`MLflow Tracking Servers <https://mlflow.org/docs/latest/tracking.html#mlflow-tracking-servers>`__ to get you started.
In order to use the MLflow Model Registry, you will need to setup an MLflow Tracking Server with a database Backend
Store, such as SQLite or PostgreSQL.

