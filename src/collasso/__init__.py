"""Sparse linear multi-task regression"""

__version__ = "0.0.1"
__docformat__ = "numpy"

from .functions import (
    _simulate_features,
    _simulate_effects,
    _simulate_targets,
    simulate,
    SingleTaskLassoCV,
    CoopLasso,
    CoopLassoCV,
    _calc_weights_slow,
    _calc_weights_fast,
    _spearmanr,
)

__all__ = [
    "_simulate_features",
    "_simulate_effects",
    "_simulate_targets",
    "simulate",
    "SingleTaskLassoCV",
    "CoopLasso",
    "CoopLassoCV",
    "_calc_weights_slow",
    "_calc_weights_fast",
    "_spearmanr"
]
