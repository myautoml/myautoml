import importlib
import logging
from pathlib import Path
import requests
import tempfile
from yaml import safe_load

from box import Box
import joblib
import mlflow
import mlflow.sklearn
from mlflow.entities import RunInfo

_logger = logging.getLogger(__name__)


def get_model(run_id: str,
              model_path: str = "model"):
    # Determine how to load the model
    ml_model_url = f"{mlflow.get_tracking_uri()}/get-artifact?path={model_path}%2FMLmodel&run_id={run_id}"
    ml_model = Box(safe_load(requests.get(ml_model_url).content))
    loader_module = ml_model.flavors.python_function.loader_module
    _logger.debug(f"Loader module for the model: {loader_module}")

    # Import the 'load_model' function
    load_model = getattr(importlib.import_module(loader_module), 'load_model')

    # Load the model
    model = load_model(f"runs:/{run_id}/{model_path}")
    return model


def track_model_from_file(local_path: str,
                          artifact_path: str = 'model',
                          experiment_name: str = 'default',
                          run_name: str = None) -> RunInfo:
    _logger.debug("Uploading model to MLflow Server")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_artifact(local_path=local_path, artifact_path=artifact_path)
        run_info = run.info
    return run_info


def track_model(model,
                artifact_path: str = 'model',
                experiment_name: str = 'default',
                run_name: str = None) -> RunInfo:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / 'model.joblib'
        _logger.debug(f"Saving model to temporary file: {temp_file}")
        joblib.dump(model, temp_file)
        run_info = track_model_from_file(local_path=temp_file,
                                         artifact_path=artifact_path,
                                         experiment_name=experiment_name,
                                         run_name=run_name)
    return run_info


def track_model_data(run_id: str,
                     params: dict = None,
                     metrics: dict = None,
                     tags: dict = None,
                     artifacts: dict = None,
                     artifact_dirs: dict = None) -> None:
    _logger.debug(f"Tracking model metadata for run_id: {run_id}")
    client = mlflow.tracking.MlflowClient()
    if params is not None:
        for key, value in params.items():
            client.log_param(run_id, key, value)
    if metrics is not None:
        for key, value in metrics.items():
            client.log_metric(run_id, key, value)
    if tags is not None:
        for key, value in tags.items():
            client.set_tag(run_id, key, value)
    if artifacts is not None:
        for local_path, artifact_path in artifacts.items():
            client.log_artifact(run_id, local_path, artifact_path)
    if artifact_dirs is not None:
        for local_dir, artifact_path in artifact_dirs.items():
            client.log_artifacts(run_id, local_dir, artifact_path)


def log_sk_model(sk_model, registered_model_name=None, params=None, tags=None, metrics=None, artifacts=None,
                 artifact_path='model'):
    _logger.info("Logging Scikit-Learn model to MLflow")
    mlflow.sklearn.log_model(sk_model=sk_model,
                             artifact_path=artifact_path,
                             conda_env='./environment.yml',
                             registered_model_name=registered_model_name)
    mlflow.log_params(params)
    mlflow.set_tags(tags)
    mlflow.log_metrics(metrics)
    for local_path, artifact_path in artifacts.items():
        _logger.debug(f"Logging artifact to MLflow: {local_path} - {artifact_path}")
        mlflow.log_artifact(local_path, artifact_path)
