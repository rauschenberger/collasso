"""Sparse linear multi-task regression"""

__version__ = "0.0.1"
__docformat__ = "numpy"

from collasso.simulate import (
    simulate,
    _simulate_features,
    _simulate_effects,
    _simulate_targets,
    )
from collasso.single_task import SingleTaskLassoCV
from collasso.multi_task import (
    CoopLasso,
    CoopLassoCV,
    _spearmanr,
    _calc_weights_slow,
    _calc_weights_fast,
    )

__all__ = [
    "simulate",
    "_simulate_features",
    "_simulate_effects",
    "_simulate_targets",
    "SingleTaskLassoCV",
    "CoopLassoCV",
    "CoopLasso",
    "_spearmanr",
    "_calc_weights_slow",
    "_calc_weights_fast",
]
