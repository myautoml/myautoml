import logging.config
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, \
    recall_score

from myautoml.utils import load_config
from myautoml.utils.mlflow import get_registered_model

from data import load_prediction_data

_logger = logging.getLogger(__file__)


def main():
    load_dotenv('.env.general')
    config = load_config('config.yml')
    Path(config.logging.handlers.debug_file_handler.filename).parent.mkdir(parents=True, exist_ok=True)
    Path(config.logging.handlers.info_file_handler.filename).parent.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(config.logging)

    _logger.info("Loading the data")
    x, y_true = load_prediction_data()

    _logger.info("Loading the model")
    model, model_version = get_registered_model(config.model.name, config.prediction.stage)

    _logger.info("Making predictions")
    y_pred = model.predict(x)
    y_pred_proba = model.predict_proba(x)[:, 1]

    _logger.info("Checking predictions")
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }
    for k, v in metrics.items():
        _logger.info(f"{k}: {v}")

    return (x, y_true, y_pred, y_pred_proba), model, metrics


if __name__ == "__main__":
    np.random.seed(1)
    output = main()
    _logger.info("Done")
