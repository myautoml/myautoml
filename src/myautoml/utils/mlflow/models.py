import logging
from pathlib import Path

import mlflow

from .tracking import get_model

_logger = logging.getLogger(__name__)


def get_registered_model(model_name: str,
                         model_stage: str):
    _logger.debug("Finding trained model on MLflow Server")
    _logger.debug(f"Model name: {model_name}")
    _logger.debug(f"Model stage: {model_stage}")

    client = mlflow.tracking.MlflowClient()
    registered_models = client.search_model_versions(f"name='{model_name}'")

    registered_models_in_stage = [rm for rm in registered_models
                                  if rm.current_stage == model_stage]

    if len(registered_models_in_stage) == 0:
        raise LookupError(f"{model_stage} version of model {model_name} not found")
    if len(registered_models_in_stage) > 1:
        _logger.warning(f"{len(registered_models_in_stage)} {model_stage} versions of model {model_name} found")

    rm = registered_models_in_stage[-1]
    _logger.info(f"Loading model '{model_name}' version {rm.version}")

    model_path = None
    for p in Path(rm.source).parents:
        if p.stem == 'artifacts':
            model_path = Path(rm.source).relative_to(p)

    if model_path is None:
        raise ValueError("Path of model in artifact store not found")

    return get_model(rm.run_id, str(model_path)), rm.version


def register_model(run_id: str,
                   model_name: str,
                   artifact_path: str = 'model'):
    _logger.debug("Registering model in MLflow Server")
    model_version = mlflow.register_model(
        f"runs:/{run_id}/artifacts/{artifact_path}",
        model_name
    )
    return model_version
