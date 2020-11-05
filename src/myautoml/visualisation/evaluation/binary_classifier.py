import logging
from pathlib import Path

import matplotlib.pyplot as plt

from myautoml.visualisation.colors import EVALUATION_COLORS, TRAIN_COLOR, TEST_COLOR

from . import plot_roc, plot_cum_precision, plot_lift_deciles, plot_precision_recall, plot_prediction_distribution, \
    plot_calibration_curve, plot_calibration_curve_zoom

_logger = logging.getLogger(__name__)


def save_roc_curve(save_dir, data):
    _logger.debug("Plotting the ROC curve")
    save_path = Path(save_dir) / 'roc.png'
    fig, ax = plt.subplots()
    try:
        for label in data.keys():
            plot_roc(ax, data[label]['y'], data[label]['y_pred_proba'],
                     label=label, color=EVALUATION_COLORS[label])
        fig.savefig(save_path)
    except Exception as e:
        _logger.warning(f"Error plotting the ROC curve: {str(e)}")
        save_path = None
    finally:
        plt.close(fig)
    return save_path


def save_precision_recall_curve(save_dir, data):
    _logger.debug("Plotting the Precision-Recall curve")
    save_path = Path(save_dir) / 'precision_recall.png'
    fig, ax = plt.subplots()
    try:
        for label in data.keys():
            plot_precision_recall(ax, data[label]['y'], data[label]['y_pred_proba'],
                                  label=label, color=EVALUATION_COLORS[label],
                                  legend_loc='upper right')
        fig.savefig(save_path)
    except Exception as e:
        _logger.warning(f"Error plotting the Precision-Recall curve: {str(e)}")
        save_path = None
    finally:
        plt.close(fig)
    return save_path


def save_lift_deciles(save_dir, data):
    _logger.debug("Plotting the lift deciles")
    save_path = Path(save_dir) / 'lift_deciles.png'
    fig, ax = plt.subplots()
    try:
        plot_lift_deciles(ax, data['test']['y'], data['test']['y_pred_proba'])
        fig.savefig(save_path)
    except Exception as e:
        _logger.warning(f"Error plotting the lift deciles: {str(e)}")
        save_path = None
    finally:
        plt.close(fig)
    return save_path


def save_cum_precision(save_dir, data):
    _logger.debug("Plotting the cumulative precision curve")
    save_path = Path(save_dir) / 'cum_precision.png'
    fig, ax = plt.subplots()
    try:
        for label in data.keys():
            plot_cum_precision(ax, data[label]['y'], data[label]['y_pred_proba'],
                               label=label, color=EVALUATION_COLORS[label])
        fig.savefig(save_path)
    except Exception as e:
        _logger.warning(f"Error plotting the cumulative precision curve: {str(e)}")
        save_path = None
    finally:
        plt.close(fig)
    return save_path


def save_prediction_distribution(save_dir, data):
    _logger.debug("Plotting the prediction distribution")
    save_path = Path(save_dir) / 'prediction_distribution.png'
    fig, ax = plt.subplots()
    try:
        if 'train' in data.keys():
            plot_prediction_distribution(ax, (data['train']['y_pred_proba'], data['test']['y_pred_proba']),
                                         label=('train', 'test'),
                                         color=(TRAIN_COLOR, TEST_COLOR))
        else:
            plot_prediction_distribution(ax, data['test']['y_pred_proba'],
                                         label='test',
                                         color=TEST_COLOR)
        fig.savefig(save_path)
    except Exception as e:
        _logger.warning(f"Error plotting the prediction distribution: {str(e)}")
        save_path = None
    finally:
        plt.close(fig)
    return save_path


def save_calibration_curve(save_dir, data):
    _logger.debug("Plotting the calibration curve")
    save_path = Path(save_dir) / 'calibration_curve.png'
    fig, ax = plt.subplots()
    try:
        plot_calibration_curve(ax, data['test']['y'], data['test']['y_pred_proba'],
                               label='test',
                               color=TEST_COLOR)

        fig.savefig(save_path)
    except Exception as e:
        _logger.warning(f"Error plotting the calibration curve: {str(e)}")
        save_path = None
    finally:
        plt.close(fig)
    return save_path


def save_calibration_curve_zoom(save_dir, data):
    _logger.debug("Plotting the calibration curve zoom")
    save_path = Path(save_dir) / 'calibration_curve_zoom.png'
    fig, ax = plt.subplots()
    try:
        plot_calibration_curve_zoom(ax, data['test']['y'], data['test']['y_pred_proba'],
                                    label='test',
                                    color=TEST_COLOR)

        fig.savefig(save_path)
    except Exception as e:
        _logger.warning(f"Error plotting the calibration curve zoom: {str(e)}")
        save_path = None
    finally:
        plt.close(fig)
    return save_path
