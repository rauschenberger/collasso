"""Sparse linear multi-task regression."""

__version__ = "0.1.0"
__docformat__ = "numpy"

from collasso.multi_task import (
    CoopLassoCV,
    _calc_weights,
    _CoopLasso,
    _spearmanr,
)
from collasso.simulate import (
    _simulate_effects,
    _simulate_features,
    _simulate_targets,
    simulate,
)
from collasso.single_task import IndepLassoCV

__all__ = [
    "CoopLassoCV",
    "IndepLassoCV",
    "_CoopLasso",
    "_calc_weights",
    "_simulate_effects",
    "_simulate_features",
    "_simulate_targets",
    "_spearmanr",
    "simulate",
]
