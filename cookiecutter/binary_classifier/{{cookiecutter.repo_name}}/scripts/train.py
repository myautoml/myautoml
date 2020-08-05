import logging.config
from pathlib import Path
import tempfile

from dotenv import load_dotenv
from hyperopt import fmin, tpe, Trials, STATUS_OK
import mlflow
import numpy as np

from myautoml.evaluation.binary_classifier import evaluate_binary_classifier
from myautoml.evaluation.shap import shap_analyse
from myautoml.utils import load_config
from myautoml.utils.hyperopt import flatten_params, prep_params
from myautoml.utils.mlflow import log_sk_model
from myautoml.utils.model import make_pipeline

from data import load_training_data, split_data
from model import get_preprocessor, get_estimator, get_params

_logger = logging.getLogger(__file__)


def train_run(estimator_params, x_train_prep, y_train, x_test_prep, y_test, temp_dir):
    temp_dir.mkdir(parents=True, exist_ok=True)
    _logger.info("Fitting the estimator")
    estimator, estimator_tags = get_estimator(**estimator_params)
    estimator.fit(x_train_prep, y_train)

    estimator_metrics, estimator_artifacts = evaluate_binary_classifier(
        model=estimator,
        data={'train': {'x': x_train_prep, 'y': y_train},
              'test': {'x': x_test_prep, 'y': y_test}},
        temp_dir=temp_dir)
    return estimator, estimator_tags, estimator_metrics, estimator_artifacts


def main():
    load_dotenv('.env.general')
    config = load_config('config.yml')
    Path(config.logging.handlers.debug_file_handler.filename).parent.mkdir(parents=True, exist_ok=True)
    Path(config.logging.handlers.info_file_handler.filename).parent.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(config.logging)

    _logger.info("Loading the data")
    x, y = load_training_data()
    x_train, x_test, y_train, y_test = split_data(x, y)

    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        mlflow.set_experiment(config.experiment.name)

        params = {}
        tags = {}
        metrics = {}
        artifacts = {}

        with mlflow.start_run():
            _logger.info("Fitting the preprocessor")
            preprocessor = get_preprocessor()
            preprocessor.fit(x_train, y_train)

            _logger.info("Preprocessing the training data")
            x_train_prep = preprocessor.transform(x_train)
            x_test_prep = preprocessor.transform(x_test)

            estimator_params, search_space = get_params()

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

                    loss = 1 - ho_metrics[config.evaluation.primary_metric]

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
                fmin(fn=hyperopt_objective,
                     space=search_space,
                     algo=tpe.suggest,
                     trials=trials,
                     max_evals=config.training.max_evals,
                     rstate=np.random.RandomState(1),
                     show_progressbar=False)

                model = trials.best_trial['result']['model']
                params = trials.best_trial['result']['params']
                tags = trials.best_trial['result']['tags']
                metrics = trials.best_trial['result']['metrics']
                artifacts = trials.best_trial['result']['artifacts']

            if config.evaluation.shap_analysis:
                _logger.info("Starting shap analysis")
                shap_tags, shap_artifacts = shap_analyse(model=model, x=x_train, temp_dir=Path(temp_dir) / 'shap')
                tags.update(shap_tags)
                artifacts.update(shap_artifacts)
            else:
                _logger.info("Shap analysis skipped")

            log_sk_model(model, registered_model_name=None,
                         params=params, tags=tags, metrics=metrics, artifacts=artifacts)

    return (x_train, y_train, x_test, y_test), model, params, tags, metrics, artifacts


if __name__ == "__main__":
    np.random.seed(1)
    output = main()
    _logger.info("Done")
