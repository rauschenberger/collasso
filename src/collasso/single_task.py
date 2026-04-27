"""
Single-Task Learning (convenience functions).

Class:
    ``IndepLassoCV`` - a wrapper function for ``sklearn.linear_model.LassoCV``
    to model multiple targets based on a common feature matrix
    or specific feature matrices (using the same interface as ``CoopLassoCV``)

Example
-------
    >>> from sklearn.datasets import load_linnerud
    >>> from collasso import CoopLassoCV
    >>> x, y = load_linnerud(return_X_y=True)
    >>> model = IndepLassoCV()
    >>> model.fit(x, y) # n_samples x p_features, n_samples x q_targets
    >>> model.coef_ # q_targets x p_features
    >>> y_pred = model.predict(x) # n_samples x q_targets
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LassoCV
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from collasso._helpers import (
    _check_dims,
    _format_mask,
    _validate_train_data,
    _validate_test_data,
)


class IndepLassoCV(RegressorMixin, BaseEstimator): # noqa: DOC105
    # pylint: disable=too-many-instance-attributes
    """
    Single-Task Lasso Regression For Multiple Targets.

    Fits single-task lasso regression separately to multiple targets,
    optimising the regularisation parameters by cross-validation.

    This is a convencience class with the interface as
    ``CoopLassoCV`` (but without sharing information among targets or features).

    Parameters
    ----------
    cv : int, default=10
        Number of cross-validation folds.
    alphas : int, default=100
        Number of candidate values for the regularisation parameter.

    Attributes
    ----------
    n_ : int
        Number of training samples.
    p_ : int
        Number of features.
    q_ : int
        Number of targets.
    model_ : list of length q_targets
        Fitted models from LassoCV (one for each target).
    coef_ : ndarray of shape (q_targets, p_features)
        Estimated coefficients
        (of the feature in the column on the target in the row).
        
    See Also
    --------
    CoopLassoCV
        The main class of this package.
        It uses the same interface as ``IndepLassoCV``
        (similarly formatted inputs and outputs)
        but shares information among targets and features
        to improve selection and prediction.
        
    Examples
    --------
    >>> from sklearn.datasets import load_linnerud
    >>> from collasso import IndepLassoCV
    >>> x, y = load_linnerud(return_X_y=True)
    >>> model = IndepLassoCV()
    >>> model.fit(x, y) # n_samples x p_features, n_samples x q_targets
    >>> model.coef_ # q_targets x p_features
    >>> y_pred = model.predict(x) # n_samples x q_targets
    """

    def __init__(self, *, cv: int = 10, alphas: int = 100): # noqa: DOC105
        self.cv = cv
        self.alphas = alphas
        self.n_: int
        self.p_: int
        self.q_: int
        self.n_features_in_: int
        self.model_: list
        self.coef_: np.ndarray
        self.z_: np.ndarray

    def fit( # noqa: DOC105
      self, X: np.ndarray, y: np.ndarray, Z: np.ndarray|None = None
    ) -> "IndepLassoCV":
        # pylint: disable=invalid-name
        """
        Fit IndepLassoCV.

        Parameters
        ----------
        X : ndarray of shape (n_samples, p_features) or (n_samples, p_features, q_targets)
            Common feature matrix for all targets or
            a separate feature matrix for each target.
        y : ndarray of shape (n_samples, q_targets)
            Target matrix.
        Z : ndarray of shape (p_features,) or (p_features, q_targets), or None
            Logical vector or matrix
            indicating primary (1/True)
            and auxiliary features (0/False)
            for all targets together or each target separately
            (NB: auxiliary features are simply excluded).

        Returns
        -------
        self: IndepLassoCV
            Fitted models.
        """
        X, y = _validate_train_data(self=self, X=X, y=y)
        check_array(array=X, allow_nd=True, dtype="numeric")
        check_array(array=y, dtype="numeric")
        #if X.ndim == 2:
        #    X = np.broadcast_to(X[:, :, None], (X.shape[0], X.shape[1], y.shape[1]))
        self.n_, self.p_, self.q_ = _check_dims(X=X, y=y, Z=Z)
        self.n_features_in_ = self.p_
        self.z_ = _format_mask(self,Z=Z)
        self.model_ = []
        self.coef_ = np.full((self.q_, self.p_), np.nan)
        xx = np.empty(0)
        if X.ndim == 2:
            xx = X.copy()
        for i in range(self.q_):
            if X.ndim == 3:
                xx = X[:, :, i].copy()
            xx[:, self.z_[:, i] == 0] = 0
            model = LassoCV(alphas=self.alphas, cv=self.cv)
            model.fit(xx, y[:, i])
            self.model_.append(model)
            self.coef_[i, :] = model.coef_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray: # noqa: DOC105
        # pylint: disable=invalid-name
        """
        Make predictions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, p_features) or (n_samples, p_features, q_targets)
            Common feature matrix for all targets,
            or a separate feature matrix for each target.

        Returns
        -------
        y_hat : ndarray of shape (n_samples, q_targets)
            Matrix of predicted values.
        """
        check_is_fitted(self, attributes=["coef_"])
        X = _validate_test_data(self=self, X=X)
        if X.ndim == 2:
            X = np.broadcast_to(X[:, :, None], (X.shape[0], self.p_, self.q_))
        y_hat = np.full((X.shape[0], self.q_), np.nan)
        for i in range(self.q_):
            y_hat[:, i] = self.model_[i].predict(X[:, :, i])
        if self.q_ == 1:
            return y_hat.ravel()
        return y_hat
