import logging

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from hyperopt import hp

_logger = logging.getLogger(__name__)


def get_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, selector(dtype_exclude=["category", "object"])),
        ('cat', categorical_transformer, selector(dtype_include=["category"]))]
    )
    return preprocessor


def get_estimator(**params):
    estimator = RandomForestClassifier(**params)
    estimator_tags = {'module': 'sklearn.ensemble',
                      'class': 'RandomForestClassifier'}
    return estimator, estimator_tags


def get_params():
    # estimator_params = {
    #     'n_estimators': 100,
    #     'max_depth': 5
    # }
    # search_space = None

    estimator_params = {}
    search_space = {
        'n_estimators': hp.quniform('num_leaves', 10, 150, 1),
        'max_depth': hp.quniform('max_depth', 3, 6, 1)
    }
    return estimator_params, search_space
