"""Sparse linear multi-task regression"""

__version__ = "0.0.1"
__docformat__ = "numpy"

from .functions import (
    simulate,
    SingleTaskLassoCV,
    CoopLasso,
    CoopLassoCV,
)

__all__ = ["simulate", "SingleTaskLassoCV", "CoopLasso", "CoopLassoCV"]

