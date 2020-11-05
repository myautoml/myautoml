import logging

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, \
    recall_score
from sklearn.model_selection import cross_validate

from myautoml.visualisation.evaluation.binary_classifier import (
    save_roc_curve, save_cum_precision, save_prediction_distribution, save_lift_deciles, save_precision_recall_curve,
    save_calibration_curve, save_calibration_curve_zoom)

_logger = logging.getLogger(__name__)


def get_metrics(model, data, prefix=None):
    _logger.debug(f"Starting computing the metrics")
    if prefix is None:
        prefix = ''
    elif len(prefix) > 0 and (not prefix[-1:] == "_"):
        prefix = prefix + "_"

    metrics = {}
    for label in data.keys():
        x = data[label]['x']
        y_true = data[label]['y']
        y_pred = data[label]['y_pred'] = model.predict(x)
        y_pred_proba = data[label]['y_pred_proba'] = model.predict_proba(x)[:, 1]

        metrics[f"{prefix}roc_auc_{label}"] = roc_auc_score(y_true, y_pred_proba)
        metrics[f"{prefix}average_precision_{label}"] = average_precision_score(y_true, y_pred_proba)
        metrics[f"{prefix}accuracy_{label}"] = accuracy_score(y_true, y_pred)
        metrics[f"{prefix}f1_{label}"] = f1_score(y_true, y_pred)
        metrics[f"{prefix}precision_{label}"] = precision_score(y_true, y_pred)
        metrics[f"{prefix}recall_{label}"] = recall_score(y_true, y_pred)

    return metrics


def get_plots(temp_dir, data, plots, plot_path='evaluation'):
    artifacts = {}

    # Standard evaluation plots
    if 'roc' in plots:
        roc_curve_path = save_roc_curve(temp_dir, data)
        if roc_curve_path:
            artifacts[roc_curve_path] = plot_path

    if 'pr' in plots:
        pr_curve_path = save_precision_recall_curve(temp_dir, data)
        if pr_curve_path:
            artifacts[pr_curve_path] = plot_path

    if 'lift_deciles' in plots:
        lift_deciles_path = save_lift_deciles(temp_dir, data)
        if lift_deciles_path:
            artifacts[lift_deciles_path] = plot_path

    if 'cum_precision' in plots:
        cum_precision_path = save_cum_precision(temp_dir, data)
        if cum_precision_path:
            artifacts[cum_precision_path] = plot_path

    if 'distribution' in plots:
        distribution_path = save_prediction_distribution(temp_dir, data)
        if distribution_path:
            artifacts[distribution_path] = plot_path

    # Calibration plots
    if 'curve' in plots:
        calibration_curve_path = save_calibration_curve(temp_dir, data)
        if calibration_curve_path:
            artifacts[calibration_curve_path] = plot_path

    if 'curve' in plots:
        calibration_curve_zoom_path = save_calibration_curve_zoom(temp_dir, data)
        if calibration_curve_zoom_path:
            artifacts[calibration_curve_zoom_path] = plot_path

    return artifacts


def evaluate_binary_classifier(model, data, temp_dir, plots='all'):
    _logger.debug(f"Starting evaluation for binary classifier")

    metrics = get_metrics(model, data)

    _logger.debug(f"Starting cross-validation for binary classifier")
    scorers = ['roc_auc', 'accuracy', 'f1', 'average_precision', 'precision', 'recall']
    cv_results = cross_validate(estimator=model, X=data['train']['x'], y=data['train']['y'],
                                scoring=scorers, cv=5)
    for scorer in scorers:
        metrics[f"{scorer}_cv"] = cv_results[f"test_{scorer}"].mean()

    if plots is None or plots == "":
        artifacts = {}
    else:
        if plots == 'all':
            plots = ['roc', 'pr', 'lift_deciles', 'cum_precision', 'distribution']

        artifacts = get_plots(temp_dir, data, plots, plot_path='evaluation')

    return metrics, artifacts


def evaluate_calibration(model, data, temp_dir, plots='all'):
    _logger.debug(f"Starting evaluation calibration for binary classifier")

    metrics = get_metrics(model, data, prefix='calibration')

    if plots is None or plots == "":
        artifacts = {}
    else:
        if plots == 'all':
            plots = ['curve', 'curve_zoom', 'roc', 'pr', 'lift_deciles', 'cum_precision', 'distribution']

        artifacts = get_plots(temp_dir, data, plots, plot_path='evaluation_calibration')

    return metrics, artifacts
