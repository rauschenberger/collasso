"""Sparse linear multi-task regression."""

__version__ = "0.1.0"
__docformat__ = "numpy"

from collasso.simulate import (
    simulate,
    _simulate_features,
    _simulate_effects,
    _simulate_targets,
)
from collasso.single_task import IndepLassoCV
from collasso.multi_task import (
    CoopLassoCV,
    _CoopLasso,
    _spearmanr,
    _calc_weights,
)

__all__ = [
    "simulate",
    "_simulate_features",
    "_simulate_effects",
    "_simulate_targets",
    "IndepLassoCV",
    "CoopLassoCV",
    "_CoopLasso",
    "_spearmanr",
    "_calc_weights",
]
