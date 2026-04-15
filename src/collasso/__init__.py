"""Sparse linear multi-task regression"""

__version__ = "0.0.1"
__docformat__ = "numpy"

from .functions import (
    simulate,
    SingleTaskLassoCV,
    CoopLasso,
    CoopLassoCV,
    _calc_weights_slow,
    _calc_weights_fast,
    _spearmanr,
)

__all__ = [
    "simulate",
    "SingleTaskLassoCV",
    "CoopLasso",
    "CoopLassoCV",
    "_calc_weights_slow",
    "_calc_weights_fast",
    "_spearmanr"
]
