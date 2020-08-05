import logging

from sklearn.pipeline import Pipeline

_logger = logging.getLogger(__name__)


def make_pipeline(preprocessor, estimator):
    _logger.debug("Defining the model as a pipeline")
    return Pipeline(steps=[('preprocessor', preprocessor),
                           ('estimator', estimator)])
