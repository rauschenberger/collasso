"""
Sparse linear multi-task regression
with correlation-based information sharing

Classes:
    SingleTaskLassoCV: Modelling multiple targets separately
    CoopLassoCV: Modelling multiple targets together

Example:
    # from collasso import CoopLassoCV
    # x_train, y_train, x_test, y_test, beta = simulate()
    # model = CoopLassoCV()
    # model.fit(x_train, y_train)
    # model.predict(x_test)
"""

import warnings
import numpy as np
from scipy.stats import rankdata
from sklearn.exceptions import DataConversionWarning
from sklearn.utils import check_array
from sklearn.utils.validation import validate_data


def _check_dims(
    *, X: np.ndarray, y: np.ndarray, Z: np.ndarray | None
) -> tuple[int, int, int]:
    # pylint: disable=invalid-name
    """
    Check dimensionality of inputs

    Parameters
    ----------
    X : np.ndarray of shape (n_samples,p_features) or (n_samples,p_features,q_targets)
        feature matrix
    y : np.ndarray of shape (n_samples,p_targets)
        target matrix
    Z : np.ndarray of shape (p_features) or (p_features,q_targets) or None
        indicator matrix (0=auxiliary, 1=primary)

    Raises
    ------
    ValueError

    Returns
    -------
    n : int
        number of samples
    p : int
        number of features
    q : int
        number of targets
    """
    # --- targets ---
    if y.ndim != 2:
        raise ValueError("'y' should be an 'n x q' matrix")
    n, q = y.shape

    # --- features ---
    if X.ndim not in (2, 3):
        raise ValueError("'X' should be an 'n x p' matrix or an 'n x p x q' array")
    if X.shape[0] != n:
        raise ValueError(
            "'y' and 'X' should have the same number of samples"
            "(first dimension in 'y' and 'X')"
        )
    if X.ndim == 3 and X.shape[2] != q:
        raise ValueError(
            "'y' and 'X' should have the same number of targets"
            "(second dimension in 'y', third dimension in 'X')"
        )
    p = X.shape[1]

    # --- indicators ---
    if Z is not None:
        if Z.ndim not in (1, 2):
            raise ValueError("'Z' should be a 'p' vector or an 'p x q' matrix")
        if (Z.ndim == 1 and Z.shape[0] != p) or (Z.ndim == 2 and Z.shape[0] != p):
            raise ValueError(
                "'X' and 'Z' should have the same number of features"
                "(second dimension in 'X', first dimension in 'Z')"
            )
        if Z.ndim == 2 and Z.shape[1] != q:
            raise ValueError(
                "'y' and 'Z' should have the same number of targets"
                "(second dimension in 'y' and 'Z')"
            )

    # --- edge cases ---
    if n < 2:
        raise ValueError(f"Requires more than 1 sample (now: n={n}).")
    if p < 2:
        raise ValueError(f"Requires at least 2 features (now: p={p}, n_features = 1).")
    if q < 1:
        raise ValueError(f"Requires at least 1 target (now: q={q}).")

    return n, p, q


def _spearmanr(x: np.ndarray) -> np.ndarray:
    """
    Spearman correlation coefficients

    Returns a matrix also in degenerate cases (one or two features).
    """
    if x.shape[1] == 1:
        cor = np.ones((1, 1))
    else:
        cor = np.atleast_2d(np.corrcoef(rankdata(x, axis=0), rowvar=False))
        cor = np.where(np.isnan(cor), np.eye(cor.shape[0]), cor)
        cor = np.atleast_2d(np.asarray(cor))
    return cor


def _validate_train_data(
    self, *, X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate training data when y is a vector or matrix and X is a matrix or array
    """
    if y is None:
        raise ValueError(
            "Requires target matrix y."
            "(requires y to be passed, but the target y is None)"
        )
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        warnings.warn(
            "A column-vector y was passed when a 1d array was expected.",
            DataConversionWarning,
            stacklevel=2,
        )
        y = y.ravel()
    if isinstance(X, np.ndarray) and X.ndim == 3:
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        check_array(array=X, allow_nd=True, dtype="numeric")
        check_array(array=y, dtype="numeric")
    else:
        X, y = validate_data(
            self,
            X=X,
            y=y,
            multi_output=True,
            y_numeric=True,
            dtype="numeric",
            allow_nd=True,
        )
        if y.ndim == 1:
            y = y.reshape(-1, 1)
    return X, y

def _format_mask(self, *, Z: np.ndarray|None) -> np.ndarray:
    """
    Transform Z to p x q matrix
    """
    if Z is None:
        Z = np.full((self.p_, self.q_), 1)
    elif Z.ndim == 1:
        Z = np.broadcast_to(Z[:, None], (self.p_, self.q_))
    return Z

def _validate_test_data(self, *, X: np.ndarray) -> np.ndarray:
    """
    Validate testing data X is a matrix or array
    """
    # Z = _format_mask(self,Z=self.Z)
    if isinstance(X, np.ndarray) and X.ndim == 3:
        check_array(X, allow_nd=True, dtype="numeric")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} but received {X.shape[1]} features"
            )
    else:
        X = validate_data(self, X=X, reset=False, dtype="numeric")
    return X
