import logging
from pathlib import Path

import matplotlib.pyplot as plt
import shap

_logger = logging.getLogger(__name__)


def save_shap_summary(save_dir, shap_values, shap_data, shap_feature_names):
    _logger.debug("Plotting Shap summary diagram")
    save_path = Path(save_dir) / 'shap_summary.png'
    shap.summary_plot(shap_values, shap_data, show=False, feature_names=shap_feature_names)
    fig = plt.gcf()
    fig.set_figwidth(10)
    fig.set_figheight(5)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_shap_summary_bar(save_dir, shap_values, shap_data, shap_feature_names):
    _logger.debug("Plotting Shap summary bar diagram")
    save_path = Path(save_dir) / 'shap_summary_bar.png'
    shap.summary_plot(shap_values, shap_data, plot_type='bar', show=False, feature_names=shap_feature_names)
    fig = plt.gcf()
    fig.set_figwidth(10)
    fig.set_figheight(5)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


# noinspection DuplicatedCode
def save_shap_dependence_plots(save_dir, shap_values, shap_data, shap_feature_names):
    # Plot dependence plots for each variable in the model
    save_paths = []
    for col in shap_feature_names:
        _logger.debug(f"Plotting dependence plot for {col}")
        save_path = Path(save_dir) / f"shap_dependence_{col}.png"
        shap.dependence_plot(ind=col,
                             shap_values=shap_values,
                             features=shap_data,
                             interaction_index=col,
                             show=False,
                             feature_names=shap_feature_names)
        fig = plt.gcf()
        fig.set_figwidth(10)
        fig.set_figheight(5)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        save_paths.append(save_path)
    return save_paths
