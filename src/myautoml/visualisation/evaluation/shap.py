import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype, is_bool_dtype
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
def save_shap_dependence_plots(save_dir, shap_values, shap_data, shap_feature_names, display_features, preprocessor=None):
    # Plot dependence plots for each variable in the model
    save_paths = []
    for col in shap_feature_names:
        _logger.debug(f"Plotting dependence plot for {col}")
        save_path = Path(save_dir) / f"shap_dependence_{col}.png"
        shap.dependence_plot(ind=col,
                             shap_values=shap_values,
                             features=shap_data,
                             display_features=display_features[shap_feature_names],
                             # interaction_index=col,
                             interaction_index=None,
                             show=False,
                             feature_names=shap_feature_names)
        fig = plt.gcf()
        fig.set_figwidth(10)
        fig.set_figheight(5)

        # TODO: Check the following hacky solution to fix the x-axis labels in terms of the original data
        if is_numeric_dtype(display_features[col]):
            min_val = display_features[col].min()
            max_val = display_features[col].max()
            if is_bool_dtype(min_val) and min_val:
                min_val = 1
            elif is_bool_dtype(min_val) and not min_val:
                min_val = 0
            else:
                min_val = min_val

            if is_bool_dtype(max_val) and max_val:
                max_val = 1
            elif is_bool_dtype(max_val) and not max_val:
                max_val = 0
            else:
                max_val = max_val

            diff = max_val - min_val
            d = diff
            factors = 0
            while d > 10:
                d = d / 10
                factors = factors + 1

            min_tick = (10 ** factors) * math.ceil(min_val / (10 ** factors))
            max_tick = (10 ** factors) * math.floor(max_val / (10 ** factors))

            ticks_values = [x for x in range(min_tick, max_tick + 1, 10 ** factors)]

            # TODO: This specific solution for age should be dealt with differently
            if col == 'age':
                ticks_values = [x for x in range(min_tick, max_tick + 1, 10)]
            # df_ticks_values = pd.DataFrame(ticks_values, columns=[col])

            df_ticks_values = display_features.head(len(ticks_values)).copy()
            df_ticks_values[col] = ticks_values

            ticks = preprocessor.transform(df_ticks_values)
            col_idx = shap_feature_names.index(col)
            ticks = ticks[:, col_idx]

            plt.xticks(ticks, ticks_values)

        plt.axhline(y=0, color='black', linestyle='--', alpha=0.4)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        save_paths.append(save_path)
    return save_paths
