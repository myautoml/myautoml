# {{cookiecutter.project_name}}

{{cookiecutter.description}}


### Getting started

This repository allows a data scientist to
- train a binary classifier model,
- log the trained model to an MLflow server
- make predictions with a trained model

We assume the MLflow server has been installed and configured,
as well as the Conda environment you need to run your scripts.
You should created your own `.env.general` file based on the template provided,
and modify the settings conform your environment.


### Repository Structure

    ├── README.md           <- The top-level README for developers using this project
    │
    ├── scripts             <- Script files
    │   ├── .env.general    <- The environment file which you must configure
    │   ├── config.yml      <- The configuration file for your experiment
    │   ├── environment.yml <- The environment file defining the Conda environment for your script
    │   ├── data.py         <- Contains the functions that load your data
    │   ├── model.py        <- Contains the functions that set up your model
    │   ├── predict.py      <- Make predictions
    │   └── train.py        <- Train the model
    │
    └── src                 <- Source code for use in this project

<p><small>Project based on the MyAutoML Cookiecutter template for
<a target="_blank" href="https://github.com/erikjandevries/myautoml/tree/master/cookiecutter/binary_classifier">
binary classifiers</a>.</small></p>
