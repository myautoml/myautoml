import logging

import shap

from myautoml.utils.sklearn import get_ct_feature_names
from myautoml.visualisation.evaluation.shap import (
    save_shap_summary, save_shap_dependence_plots, save_shap_summary_bar)

_logger = logging.getLogger(__name__)


def shap_analyse(model, x, temp_dir, return_shap_details=False):
    temp_dir.mkdir(parents=True, exist_ok=True)
    _logger.debug("Performing Shap analysis")
    shap_feature_names = get_ct_feature_names(model.steps[0][1])

    shap_estimator = model.steps[1][1]
    shap_data = model.steps[0][1].transform(x)

    shap_explainer = shap.TreeExplainer(shap_estimator)
    shap_values = shap_explainer.shap_values(shap_data)[1]

    # Computes the baseline value, i.e. expected average shape value for all customers
    tags = {
        'shap_expected_value': shap_explainer.expected_value[1]
    }

    shap_summary_path = save_shap_summary(temp_dir, shap_values, shap_data, shap_feature_names)
    shap_summary_bar_path = save_shap_summary_bar(temp_dir, shap_values, shap_data, shap_feature_names)
    shap_dependence_paths = save_shap_dependence_plots(temp_dir, shap_values, shap_data, shap_feature_names, x,
                                                       model.steps[0][1])

    paths = [shap_summary_path, shap_summary_bar_path, *shap_dependence_paths]
    artifacts = {path: 'shap' for path in paths}

    if return_shap_details:
        return tags, artifacts, shap_explainer, shap_data, shap_values, shap_feature_names

    return tags, artifacts
