import logging.config
from pathlib import Path
import tempfile

import mlflow
from sklearn.calibration import CalibratedClassifierCV

from myautoml.evaluation.binary_classifier import evaluate_calibration
from myautoml.utils.mlflow import log_sk_model, get_model
from myautoml.utils.model import make_pipeline

_logger = logging.getLogger(__file__)


def calibrate_model(run_id, x, y):
    with tempfile.TemporaryDirectory() as td:
        _logger.debug(f"Creating temporary directory: '{td}'")
        temp_dir = Path(td)

        params = {}
        tags = {}
        metrics = {}
        artifacts = {}

        _logger.info("Loading the model")
        model = get_model(run_id, model_path='model')

        with mlflow.start_run(run_id):
            _logger.info("Preprocessing the training data")
            preprocessor = model.steps[0][1]
            x_prep = preprocessor.transform(x)

            # calibrate model
            calibrated_estimator = CalibratedClassifierCV(model.steps[1][1], cv='prefit')
            calibrated_estimator.fit(x_prep, y)

            # evaluate the model
            estimator_metrics, estimator_artifacts = evaluate_calibration(
                model=calibrated_estimator,
                data={'test': {'x': x_prep, 'y': y}},
                temp_dir=temp_dir)

            estimator_params = {}
            estimator_tags = {'calibrated': True}

            calibrated_model = make_pipeline(preprocessor, calibrated_estimator)
            params.update({f"estimator_{k}": v for k, v in estimator_params.items()})
            tags.update({f"estimator_{k}": v for k, v in estimator_tags.items()})
            metrics.update(estimator_metrics)
            artifacts.update(estimator_artifacts)

            log_sk_model(calibrated_model,
                         registered_model_name=None,
                         params=params,
                         tags=tags,
                         metrics=metrics,
                         artifacts=artifacts,
                         model_artifact_path='model_calibrated')

    return (x, y), model, params, tags, metrics, artifacts
