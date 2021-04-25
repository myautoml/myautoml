import logging.config

import numpy as np

from hyperopt import hp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from myautoml.utils import start_script
from myautoml.train import train_binary_classifier

from data import load_training_data

_logger = logging.getLogger(__file__)


def main():
    config = start_script(dotenv='.env.general',
                          config_yaml="titanic_config.yml")

    _logger.info("Loading the data")
    x, y = load_training_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    _logger.info("Define preprocessor")
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, ['age', 'body', 'fare', 'parch', 'pclass', 'sibsp']),
        ('cat', categorical_transformer, ['embarked', 'sex'])]
        # ('num', numeric_transformer, selector(dtype_exclude=["category", "object"])),
        # ('cat', categorical_transformer, selector(dtype_include=["category"]))]
        # , remainder='passthrough'
    )

    _logger.info("Define estimator parameters and hyperparameters")
    estimator_params = {}
    search_space = {
        'n_estimators': hp.quniform('num_leaves', 10, 150, 1),
        'max_depth': hp.quniform('max_depth', 3, 6, 1)
    }

    model_training, model_calibration = train_binary_classifier(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        preprocessor=preprocessor,
        estimator_cls=RandomForestClassifier,
        estimator_params=estimator_params,
        search_space=search_space,
        primary_metric=config.evaluation.primary_metric,
        max_evals=config.training.max_evals,
        shap_analysis=config.evaluation.shap_analysis,
        calibrate=config.calibration.calibrate,
        experiment_name=config.experiment.name
    )

    return model_training, model_calibration


if __name__ == "__main__":
    np.random.seed(1)
    output = main()
    _logger.info("Done")
