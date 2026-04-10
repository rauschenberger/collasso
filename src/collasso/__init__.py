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

__pdoc__ = {
    "SingleTaskLassoCV.set_score_request": False,
    "CoopLasso.set_score_request": False,
    "CoopLassoCV.set_score_request": False,
    "SingleTaskLassoCV.set_fit_request": False,
    "CoopLasso.set_fit_request": False,
    "CoopLassoCV.set_fit_request": False,
    "SingleTaskLassoCV.set_predict_request": False,
    "CoopLasso.set_predict_request": False,
    "CoopLassoCV.set_predict_request": False,
}
