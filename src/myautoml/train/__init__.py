import logging.config
from pathlib import Path
import tempfile

from hyperopt import fmin, tpe, Trials, STATUS_OK
import inspect
import mlflow
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from myautoml.evaluation.binary_classifier import evaluate_binary_classifier
from myautoml.evaluation.shap import shap_analyse
from myautoml.utils.hyperopt import flatten_params, prep_params
from myautoml.utils.mlflow import log_sk_model
from myautoml.utils.model import make_pipeline
from myautoml.calibration import calibrate_model

_logger = logging.getLogger(__file__)


def train_binary_classifier(x_train: pd.DataFrame,
                            y_train: pd.DataFrame,
                            x_test: pd.DataFrame = None,
                            y_test: pd.DataFrame = None,
                            preprocessor=None,
                            estimator_cls=DecisionTreeClassifier,
                            estimator_params=None,
                            search_space=None,
                            primary_metric: str = 'roc_auc_cv',
                            max_evals: int = 5,
                            shap_analysis: bool = False,
                            calibrate: bool = False,
                            experiment_name: str = "Default"):
    def train_run(estimator_params, x_train_prep, y_train, x_test_prep, y_test, temp_dir):
        temp_dir.mkdir(parents=True, exist_ok=True)
        _logger.info("Fitting the estimator")
        estimator = estimator_cls(**estimator_params)
        estimator_tags = {'module': inspect.getmodule(estimator).__name__,
                          'class': estimator.__class__.__name__}
        estimator.fit(x_train_prep, y_train)

        estimator_metrics, estimator_artifacts = evaluate_binary_classifier(
            model=estimator,
            data={'train': {'x': x_train_prep, 'y': y_train},
                  'test': {'x': x_test_prep, 'y': y_test}},
            temp_dir=temp_dir)

        return estimator, estimator_tags, estimator_metrics, estimator_artifacts

    with tempfile.TemporaryDirectory() as td:
        _logger.debug(f"Creating temporary directory: '{td}'")
        temp_dir = Path(td)
        _logger.debug(f"Setting MLflow experiment: '{experiment_name}'")
        mlflow.set_experiment(experiment_name)

        params = {}
        tags = {}
        metrics = {}
        artifacts = {}

        _logger.debug("Starting the MLflow run")
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            _logger.info("Fitting the preprocessor")
            if preprocessor is None:
                x_train_prep = x_train
                x_test_prep = x_test
            else:
                preprocessor.fit(x_train, y_train)

                _logger.info("Preprocessing the training data")
                x_train_prep = preprocessor.transform(x_train)
                x_test_prep = preprocessor.transform(x_test)

            if search_space is None:
                estimator, estimator_tags, estimator_metrics, estimator_artifacts = train_run(
                    estimator_params=estimator_params,
                    x_train_prep=x_train_prep,
                    y_train=y_train,
                    x_test_prep=x_test_prep,
                    y_test=y_test,
                    temp_dir=temp_dir)

                model = make_pipeline(preprocessor, estimator)
                params.update({f"estimator_{k}": v for k, v in estimator_params.items()})
                tags.update({f"estimator_{k}": v for k, v in estimator_tags.items()})
                metrics.update(estimator_metrics)
                artifacts.update(estimator_artifacts)

            else:
                def hyperopt_objective(search_params):
                    # This function is called for each set of hyper-parameters being tested by HyperOpt.
                    run_name = str(len(trials) - 1)
                    ho_params = {}
                    ho_tags = {}
                    ho_metrics = {}
                    ho_artifacts = {}

                    search_params = flatten_params(search_params)
                    search_params = prep_params(search_params)
                    ho_estimator_params = estimator_params.copy()
                    ho_estimator_params.update(search_params)

                    with mlflow.start_run(nested=True, run_name=run_name):
                        try:
                            ho_estimator, ho_estimator_tags, ho_estimator_metrics, ho_estimator_artifacts = train_run(
                                estimator_params=ho_estimator_params,
                                x_train_prep=x_train_prep,
                                y_train=y_train,
                                x_test_prep=x_test_prep,
                                y_test=y_test,
                                temp_dir=temp_dir / run_name)

                            ho_model = make_pipeline(preprocessor, ho_estimator)
                            ho_params.update({f"estimator_{k}": v for k, v in ho_estimator_params.items()})
                            ho_tags.update({f"estimator_{k}": v for k, v in ho_estimator_tags.items()})
                            ho_metrics.update(ho_estimator_metrics)
                            ho_artifacts.update(ho_estimator_artifacts)

                            ho_tags['hyperopt'] = True

                            log_sk_model(ho_model, registered_model_name=None,
                                         params=ho_params, tags=ho_tags, metrics=ho_metrics, artifacts=ho_artifacts)

                            loss = 1 - ho_metrics[primary_metric]

                        except KeyboardInterrupt:
                            mlflow.set_tag("UserInterrupted", True)
                            raise KeyboardInterrupt

                        return {
                            'loss': loss,
                            'status': STATUS_OK,
                            'model': ho_model,
                            'params': ho_params,
                            'tags': ho_tags,
                            'metrics': ho_metrics,
                            'artifacts': ho_artifacts
                        }

                trials = Trials()
                try:
                    fmin(fn=hyperopt_objective,
                         space=search_space,
                         algo=tpe.suggest,
                         trials=trials,
                         max_evals=max_evals,
                         rstate=np.random.RandomState(1),
                         show_progressbar=False)
                except KeyboardInterrupt:
                    _logger.warning("User interrupted hyperopt optimisation")

                model = trials.best_trial['result']['model']
                params = trials.best_trial['result']['params']
                tags = trials.best_trial['result']['tags']
                metrics = trials.best_trial['result']['metrics']
                artifacts = trials.best_trial['result']['artifacts']

            if shap_analysis:
                _logger.info("Starting shap analysis")
                shap_tags, shap_artifacts = shap_analyse(model=model, x=x_train, temp_dir=Path(temp_dir) / 'shap')
                tags.update(shap_tags)
                artifacts.update(shap_artifacts)
            else:
                _logger.info("Shap analysis skipped")

            log_sk_model(model, registered_model_name=None,
                         params=params, tags=tags, metrics=metrics, artifacts=artifacts)

        if calibrate:
            model_calibration = calibrate_model(run_id, x_test, y_test)
        else:
            model_calibration = ()

    model_training = ((x_train, y_train, x_test, y_test), model, params, tags, metrics, artifacts)

    return model_training, model_calibration
