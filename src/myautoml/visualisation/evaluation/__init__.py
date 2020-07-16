import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from myautoml.visualisation.colors import TEST_COLOR, BASELINE_COLOR

_logger = logging.getLogger(__name__)


def plot_roc(ax, y, y_pred_proba, legend_loc='best', label=None, *args, **kwargs):
    auc = roc_auc_score(y, y_pred_proba)
    fpr, tpr, threshold = roc_curve(y, y_pred_proba)

    ax.set_title('Receiver Operating Characteristic Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    # Baseline / Random predictions
    ax.plot([0, 1], [0, 1], linestyle='dotted', color=BASELINE_COLOR)

    if label is None:
        plot_label = "area = {:0.2f}".format(auc)
    else:
        plot_label = "{}: area = {:0.2f}".format(label, auc)

    ax.plot(fpr, tpr, label=plot_label, *args, **kwargs)
    ax.legend(loc=legend_loc)
    return ax


def plot_precision_recall(ax, y, y_pred_proba, legend_loc='best', label=None, *args, **kwargs):
    auc = average_precision_score(y, y_pred_proba)
    precision, recall, threshold = precision_recall_curve(y, y_pred_proba)

    ax.set_title('Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    if label is None:
        plot_label = "area = {:0.2f}".format(auc)
    else:
        plot_label = "{}: area = {:0.2f}".format(label, auc)

    ax.plot(recall, precision, label=plot_label, *args, **kwargs)
    ax.legend(loc=legend_loc)
    return ax


def plot_lift_deciles(ax, y, y_pred_proba):
    # Sorting the class 1 probabilities and labels
    df = pd.DataFrame({'labels': y, 'pred': y_pred_proba})
    df = df.sort_values(by=['pred'], ascending=False)

    # Calculating baseline for class 1 occurrence
    baseline = df['labels'].sum() / len(df['labels'])

    # Binning into deciles and calculating actual class 1 percentages
    df_bins = df.groupby(pd.qcut(df['pred'], 10, duplicates='drop')).apply(
        lambda a: a['labels'].sum() / len(a['labels']))
    df_bins = (df_bins / baseline)

    # Plotting the chart
    ax.set_title('Non-Cumulative Lift')

    x = np.arange(len(df_bins)) + 1
    y = np.flip(df_bins.values)
    bars = ax.bar(x, y, color=TEST_COLOR)

    ax.axhline(y=1, color=BASELINE_COLOR, linestyle='dotted')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02, round(height, 2), ha='center', va='bottom')

    return ax


# noinspection SpellCheckingInspection
def _get_y_lims_for_range(r):
    d = (r[1] - r[0]) * 0.05
    return [r[0] - d, r[1] + d]


def plot_cum_precision(ax, y, y_pred_proba, label=None, color=TEST_COLOR):
    # Sorting the class 1 probabilities and labels
    df = pd.DataFrame({'labels': y, 'pred': y_pred_proba})
    df = df.sort_values(by=['pred'], ascending=False)

    # Calculating baseline for class 1 occurrence
    baseline = df['labels'].sum() / len(df['labels'])

    # Computing lift curve
    df['cumsum'] = df['labels'].cumsum()
    df['one'] = 1
    df['rank'] = df['one'].cumsum()
    df['fraction'] = df['rank'] / len(df['labels'])
    df['lift'] = df['cumsum'] / df['rank']

    # Plotting the graph
    ax.set_title('Cumulative Precision')
    ax.plot(df['fraction'], df['lift'], label=label, color=color)
    ax.axhline(y=baseline, color=BASELINE_COLOR, linestyle='dotted')
    ax.legend(loc="upper right")

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    return ax


def plot_sorted_probabilities(ax, y, y_pred_proba, color=TEST_COLOR, ascending=True, label=None):
    # Sorting the class 1 probabilities and labels
    df = pd.DataFrame({'labels': y, 'pred_proba': y_pred_proba})
    df = df.sort_values(by=['pred_proba'], ascending=ascending)

    df['one'] = 1
    df['rank'] = df['one'].cumsum()
    df['fraction'] = df['rank'] / len(df['labels'])

    # Plotting the graph
    ax.set_title('Sorted probabilities')
    ax.plot(df['fraction'], df['pred_proba'], color=color, label=label)
    ax.legend(loc="upper center")
    return ax


def plot_prediction_distribution(ax, y_pred_proba, *args, **kwargs):
    ax.hist(y_pred_proba, bins=20, density=True, *args, **kwargs)
    ax.legend(loc='upper right')
    return ax
