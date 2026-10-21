"""Train and run duration predictors for deduplicated speech units."""

from gslm2.duration.config import DurationPredictorConfig
from gslm2.duration.model import DurationPredictor

__all__ = ["DurationPredictor", "DurationPredictorConfig"]
