.. _environment:

===========
Environment
===========

To get the most out of MyAutoML you will need to install and setup several components in your environment for MyAutoML
to work with. Please have a look at the :ref:`ml_process` to see where these components fit in.


.. _model-registry-mlflow:

Model Registry: MLflow
~~~~~~~~~~~~~~~~~~~~~~

As a model registry we work with `MLflow <https://mlflow.org>`__. MLflow has two separate modules helping us to keep
a good record of our models:

- MLflow Tracking
- MLflow Model Registry

In the :ref:`ml_process`, when we refer to a Model Registry, we mean both of these MLflow components above: every
trained model is tracked in the MLflow Tracking Server. Additionally, some will be registered with a registered model
name in the MLflow Model Registry. In the prediction process, a model is loaded from the MLflow Model Registry.

Please refer to the `installation instructions <https://mlflow.org/docs/latest/quickstart.html#installing-mlflow>`__ and
`MLflow Tracking Servers <https://mlflow.org/docs/latest/tracking.html#mlflow-tracking-servers>`__ to get you started.
In order to use the MLflow Model Registry, you will need to setup an MLflow Tracking Server with a database Backend
Store, such as SQLite or PostgreSQL.

.. toctree::
    :maxdepth: 2
    :hidden:
