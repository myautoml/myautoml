import logging

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, \
    recall_score
from sklearn.model_selection import cross_validate

from myautoml.visualisation.evaluation.binary_classifier import (
    save_roc_curve, save_cum_precision, save_prediction_distribution, save_lift_deciles, save_precision_recall_curve, save_calibration_curve)

_logger = logging.getLogger(__name__)


def evaluate_binary_classifier(model, data, temp_dir, plots='all'):
    _logger.debug(f"Starting evaluation for binary classifier")
    metrics = {}
    for label in data.keys():
        x = data[label]['x']
        y_true = data[label]['y']
        y_pred = data[label]['y_pred'] = model.predict(x)
        y_pred_proba = data[label]['y_pred_proba'] = model.predict_proba(x)[:, 1]

        metrics[f"roc_auc_{label}"] = roc_auc_score(y_true, y_pred_proba)
        metrics[f"average_precision_{label}"] = average_precision_score(y_true, y_pred_proba)
        metrics[f"accuracy_{label}"] = accuracy_score(y_true, y_pred)
        metrics[f"f1_{label}"] = f1_score(y_true, y_pred)
        metrics[f"precision_{label}"] = precision_score(y_true, y_pred)
        metrics[f"recall_{label}"] = recall_score(y_true, y_pred)

    _logger.debug(f"Starting cross-validation for binary classifier")
    scorers = ['roc_auc', 'accuracy', 'f1', 'average_precision', 'precision', 'recall']
    cv_results = cross_validate(estimator=model, X=data['train']['x'], y=data['train']['y'],
                                scoring=scorers, cv=5)
    for scorer in scorers:
        metrics[f"{scorer}_cv"] = cv_results[f"test_{scorer}"].mean()

    artifacts = {}

    if not (plots is None or plots == ""):
        if plots == 'all':
            plots = ['roc', 'pr', 'lift_deciles', 'cum_precision', 'distribution']

        if 'roc' in plots:
            roc_curve_path = save_roc_curve(temp_dir, data)
            artifacts[roc_curve_path] = 'evaluation'

        if 'pr' in plots:
            pr_curve_path = save_precision_recall_curve(temp_dir, data)
            artifacts[pr_curve_path] = 'evaluation'

        if 'lift_deciles' in plots:
            lift_deciles_path = save_lift_deciles(temp_dir, data)
            artifacts[lift_deciles_path] = 'evaluation'

        if 'cum_precision' in plots:
            cum_precision_path = save_cum_precision(temp_dir, data)
            artifacts[cum_precision_path] = 'evaluation'

        if 'distribution' in plots:
            distribution_path = save_prediction_distribution(temp_dir, data)
            artifacts[distribution_path] = 'evaluation'

    return metrics, artifacts


def evaluate_calibration(model, data, temp_dir, plots='all'):
    _logger.debug(f"Starting evaluation calibration for binary classifier")

    metrics = {}
    for label in data.keys():
        x = data[label]['x']
        y_true = data[label]['y']
        y_pred = data[label]['y_pred'] = model.predict(x)
        y_pred_proba = data[label]['y_pred_proba'] = model.predict_proba(x)[:, 1]

        metrics[f"calibrated_roc_auc_{label}"] = roc_auc_score(y_true, y_pred_proba)
        metrics[f"calibrated_average_precision_{label}"] = average_precision_score(y_true, y_pred_proba)
        metrics[f"calibrated_accuracy_{label}"] = accuracy_score(y_true, y_pred)
        metrics[f"calibrated_f1_{label}"] = f1_score(y_true, y_pred)
        metrics[f"calibrated_precision_{label}"] = precision_score(y_true, y_pred)
        metrics[f"calibrated_recall_{label}"] = recall_score(y_true, y_pred)

    # _logger.debug(f"Starting cross-validation for binary classifier")
    # scorers = ['roc_auc', 'accuracy', 'f1', 'average_precision', 'precision', 'recall']
    # cv_results = cross_validate(estimator=model, X=data['test']['x'], y=data['test']['y'],
    #                             scoring=scorers, cv=5)
    # for scorer in scorers:
    #     metrics[f"{scorer}_cv"] = cv_results[f"test_{scorer}"].mean()

    artifacts = {}

    if not (plots is None or plots == ""):
        if plots == 'all':
            plots = ['curve','roc', 'pr', 'lift_deciles', 'cum_precision', 'distribution']

        # Calibration plot
        if 'curve' in plots:
            calibration_curve_path = save_calibration_curve(temp_dir, data)
            artifacts[calibration_curve_path] = 'evaluation_calibration'


        # Regular plots
        if 'roc' in plots:
            roc_curve_path = save_roc_curve(temp_dir, data)
            artifacts[roc_curve_path] = 'evaluation_calibration'

        if 'pr' in plots:
            pr_curve_path = save_precision_recall_curve(temp_dir, data)
            artifacts[pr_curve_path] = 'evaluation_calibration'

        if 'lift_deciles' in plots:
            lift_deciles_path = save_lift_deciles(temp_dir, data)
            artifacts[lift_deciles_path] = 'evaluation_calibration'

        if 'cum_precision' in plots:
            cum_precision_path = save_cum_precision(temp_dir, data)
            artifacts[cum_precision_path] = 'evaluation_calibration'

        if 'distribution' in plots:
            distribution_path = save_prediction_distribution(temp_dir, data)
            if distribution_path:
                artifacts[distribution_path] = 'evaluation_calibration'

    return metrics, artifacts

