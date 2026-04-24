"""
Multi-Task Learning

Main Class:
    ``CoopLassoCV`` - cross-validated cooperative multi-task lasso regression

Example:
    >>> from sklearn.datasets import load_linnerud
    >>> from collasso import CoopLassoCV
    >>> x, y = load_linnerud(return_X_y=True)
    >>> model = CoopLassoCV()
    >>> model.fit(x, y) # n_samples x p_features, n_samples x q_targets
    >>> model.coef_ # q_targets x p_features
    >>> y_pred = model.predict(x) # n_samples x q_targets
"""

import warnings
from typing import Union
import numpy as np
from scipy.interpolate import interp1d  # switch to np.interp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, lasso_path
from sklearn.model_selection import KFold
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from collasso._helpers import (
    _check_dims,
    _spearmanr,
    _format_mask,
    _validate_train_data,
    _validate_test_data,
)

# --- multi-task lasso regression ---

# def _construct_weights(self,*,coef,cor_y,cor_x,exp_y,exp_x,Z):
#     temp = (
#         coef *
#         (np.sign(cor_y[:, i]) *
#         (np.abs(cor_y[:, i])**self.exp_y))
#     )
#     w_pos = np.full(self.p_,np.nan)
#     w_neg = np.full(self.p_,np.nan)
#     w_abs = np.full(self.p_,np.nan)
#     for j in range(self.p_):
#         cont = (
#             temp *
#             (np.sign(cor_x[i][:, j]) *
#              (np.abs(cor_x[i][:, j])**self.exp_x))[:, np.newaxis]
#         )
#         w_pos[j] = np.sum(np.maximum(cont, 0))
#         w_neg[j] = np.sum(np.maximum(-cont, 0))
#         w_abs[j] = np.sum(np.abs(cont))
#     exclude = Z[:,i]==0
#     w_pos[exclude] = 0
#     w_neg[exclude] = 0
#     weight = np.append(w_pos+self._EPS,w_neg+self._EPS)
# return weight


def _calc_cor(*, x: np.ndarray, q: int) -> list[np.ndarray]:
    """
    Feature correlation per target.

    Calculates the Spearman correlation matrix between features for each target.
    """
    if x.ndim == 2:
        cor = _spearmanr(x)
        cor_x = [cor] * q
    elif x.ndim == 3:
        cor_x = []
        for j in range(q):
            cor = _spearmanr(x[:, :, j])
            cor_x.append(cor)
    return cor_x


def _calc_weights_slow(
    *,
    cor_y: np.ndarray,
    cor_x: np.ndarray,
    coef: np.ndarray,
    exp_y: float,
    exp_x: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptive weights
    """
    p_ = coef.shape[0]
    link_y = np.sign(cor_y) * (np.abs(cor_y) ** exp_y)
    w_pos = np.full(p_, np.nan)
    w_neg = np.full(p_, np.nan)
    w_abs = np.full(p_, np.nan)
    for j in range(p_):
        link_x = np.sign(cor_x[:, j]) * (np.abs(cor_x[:, j]) ** exp_x)
        cont = coef * link_y * link_x[:, np.newaxis]
        w_pos[j] = np.sum(np.maximum(cont, 0))
        w_neg[j] = np.sum(np.maximum(-cont, 0))
        w_abs[j] = np.sum(np.abs(cont))
    return w_pos, w_neg, w_abs


def _calc_weights_fast(
    *,
    cor_y: np.ndarray,
    cor_x: np.ndarray,
    coef: np.ndarray,
    exp_y: float,
    exp_x: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptive weights.

    Calculates adaptive weights to share information
    on feature-target effects
    between correlated features and correlated targets.
    """
    link_y = np.sign(cor_y) * np.abs(cor_y) ** exp_y
    link_x = np.sign(cor_x) * np.abs(cor_x) ** exp_x
    cont = coef * link_y[np.newaxis, :] * link_x.T[:, :, np.newaxis]
    w_pos = np.maximum(0, cont).sum(axis=(1, 2))
    w_neg = np.maximum(0, -cont).sum(axis=(1, 2))
    w_abs = np.abs(cont).sum(axis=(1, 2))
    return w_pos, w_neg, w_abs


# def _calc_weights_matrix_slow(
#     *,
#     cor_y:np.ndarray,
#     cor_x:np.ndarray,
#     coef:np.ndarray,
#     exp_y:float,
#     exp_x:float
#     ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
#     p_, q_ = coef.shape
#     w_pos = np.zeros((p_,q_))
#     w_neg = np.zeros((p_,q_))
#     w_abs = np.zeros((p_,q_))
#     for j in range(p_):
#         for k in range(q_):
#             for j_ in range(p_):
#                 for k_ in range(q_):
#                     cont = cor_x[j,j_]**exp_x*cor_y[k,k_]**exp_y*coef[j_,k_]
#                     w_pos[j,k] = w_pos[j,k] + np.maximum(0,cont)
#                     w_neg[j,k] = w_neg[j,k] + np.maximum(0,-cont)
#                     w_abs[j,k] = w_abs[j,k] + np.abs(cont)
#     return w_pos, w_neg, w_abs
#
# def _calc_weights_matrix_fast(
#     *,
#     cor_y:np.ndarray,
#     cor_x:np.ndarray,
#     coef:np.ndarray,
#     exp_y:float,
#     exp_x:float
#     ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
#     p_, q_ = coef.shape
#     w_pos = np.full((p_,q_),np.nan)
#     w_neg = np.full((p_,q_),np.nan)
#     w_abs = np.full((p_,q_),np.nan)
#     for j in range(q_):
#         link_y = np.sign(cor_y[:,j])*np.abs(cor_y[:,j])**exp_y
#         link_x = np.sign(cor_x)*np.abs(cor_x)**exp_x
#         cont = coef * link_y[np.newaxis,:] * link_x.T[:,:,np.newaxis]
#         w_pos[:,j] = np.maximum(0,cont).sum(axis=(1,2))
#         w_neg[:,j] = np.maximum(0,-cont).sum(axis=(1,2))
#         w_abs[:,j] = np.abs(cont).sum(axis=(1,2))
#     return w_pos, w_neg, w_abs


class CoopLasso(RegressorMixin, BaseEstimator):
    # pylint: disable=too-many-instance-attributes
    """
    Cooperative Multi-Task Lasso Regression.

    Fits cooperative multi-task lasso regression.

    Parameters
        ----------
    n_alphas : int, default=100
        Number of candidate values for the regularisation parameter
        in the final regressions.
    l1_ratio : float, default=0.5
        Elastic net mixing parameter for the initial regressions,
        with `0<=l1_ratio<=1`,
        where `l1_ratio=0` leads to L2 (ridge)
        and `l1_ratio=1` leads to L1 (lasso) penalisation.
    alpha_init : ndarray of shape (q_targets,) or None, default=None
        Regularisation parameters for the initial regressions,
        one non-negative number for each target
        (if `None`: optimisation by cross-validation).
    exp_y : float, default=1.0
        Non-negative number for exponentiating
        the target-target correlation coefficients.
    exp_x : float, default=1.0
        Non-negative number for exponentiating
        the feature-feature correlation coefficients.

    Attributes
    ----------
    n_ : int
        Number of training samples.
    p_ : int
        Number of features.
    q_ : int
        Number of targets.
    model_ : list
        List of `q_targets` models (one for each target)
        fitted by sklearn.linear_model.lasso_path
        with concatenated identity and inverse of feature matrix (X,-X)
        and non-negativity constraint (positive=True).
        
    See Also
    --------
    CoopLassoCV
        Optimises the regularisation parameter
        of cooperative multi-task lasso regression by cross-validation.
    """

    _EPS = 1e-09

    def __init__(
        self, *, n_alphas=100, l1_ratio=0.5, alpha_init=None, exp_y=1, exp_x=1
    ):
        # pylint: disable=too-many-arguments
        self.n_alphas = n_alphas
        self.l1_ratio = l1_ratio
        self.alpha_init = alpha_init
        self.exp_y = exp_y
        self.exp_x = exp_x
        self.n_: int
        self.p_: int
        self.q_: int
        self.n_features_in_: int
        self.mu_y_: np.ndarray
        self.sd_y_: np.ndarray
        self.alpha_init_: np.ndarray
        self.weight_: list
        self.model_: list

    def fit(
        self, X: np.ndarray, y: np.ndarray, Z: np.ndarray | None = None
    ) -> "CoopLasso":
        # pylint: disable=invalid-name,too-many-locals,too-many-branches,too-many-statements
        """
        Fit CoopLasso

        Parameters
        ----------
        X : ndarray of shape (n_samples, p_features) or (n_samples, p_features, q_targets)
            Common feature matrix for all targets
            or a specific feature matrix for each target.
        y : ndarray of shape (n_samples, q_targets)
            Target matrix.
        Z : ndarray of shape (p_features,) or (q_targets, p_features), or None
            Logical vector or matrix
            indicating primary (1, True)
            and auxiliary features (0, False)
            for all targets together or each target separately.

        Returns
        -------

        self: CoopLasso
            Fitted models.
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        check_array(array=X, allow_nd=True, dtype="numeric")
        check_array(array=y, dtype="numeric")
        self.n_, self.p_, self.q_ = _check_dims(X=X, y=y, Z=Z)
        self.n_features_in_ = self.p_
        Z = _format_mask(self,Z=Z)
        self.mu_y_ = np.mean(y, axis=0)
        self.sd_y_ = np.std(y, axis=0)
        y = (y - self.mu_y_) / self.sd_y_
        # --- calculate correlation coefficients ---
        cor_y = _spearmanr(y)
        cor_x = _calc_cor(x=X, q=self.q_)
        # --- estimate initial coefficients ---
        coef = np.full((self.p_, self.q_), np.nan)
        # if self.l1_ratio is None:
        #    raise NotImplementedError(
        #        "Initial correlation coefficients (l1_ratio=None)"
        #        "have not yet been implemented."
        #    )
        if self.alpha_init is None:
            self.alpha_init_ = np.full(self.q_, np.nan)
        else:
            self.alpha_init_ = self.alpha_init
        for j in range(self.q_):
            enet: Union[ElasticNetCV, ElasticNet]
            if self.alpha_init is None:
                enet = ElasticNetCV(l1_ratio=self.l1_ratio)
            else:
                enet = ElasticNet(alpha=self.alpha_init_[j], l1_ratio=self.l1_ratio)
            if X.ndim == 2:
                enet.fit(X, y[:, j])
            else:
                enet.fit(X[:, :, j], y[:, j])
            coef[:, j] = enet.coef_
            if self.alpha_init is None:
                assert isinstance(enet, ElasticNetCV)
                self.alpha_init_[j] = enet.alpha_
            # Alternative with multivariate initialisation:
            # enet = MultiTaskElasticNetCV(l1_ratio=l1_ratio)
            # enet.fit(X,y)
            # coef = enet.coef_.T
            # self.alpha_init_ = enet.alpha_
            # Alternative with multivariate initialisation:
            # enet = MultiTaskElasticNet(alpha=alpha_init,l1_ratio=l1_ratio)
            # enet.fit(X,y)
            # coef = enet.coef_.T
        # --- estimate final coefficients ---
        self.weight_ = []
        self.model_ = []
        xx = np.empty(0)
        if X.ndim == 2:
            xx = np.hstack([X, -X])
        for i in range(self.q_):
            if X.ndim == 3:
                xx = np.hstack([X[:, :, i], -X[:, :, i]])
            w_pos, w_neg, _ = _calc_weights_fast(
                cor_y=cor_y[:, i],
                cor_x=cor_x[i],
                coef=coef,
                exp_y=self.exp_y,
                exp_x=self.exp_x,
            )
            exclude = Z[:, i] == 0
            w_pos[exclude] = 0
            w_neg[exclude] = 0
            weight = np.append(w_pos + self._EPS, w_neg + self._EPS)
            # This alternative does not need the non-negativity constraint:
            # weight = (w_abs + 1e-9)
            xx_scale = xx * weight
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model = lasso_path(
                    X=xx_scale,
                    y=y[:, i],
                    n_alphas=self.n_alphas,
                    alphas=None,
                    positive=True,
                )
            # This alternative is computationally expensive:
            # lasso_path(X=xx_scale,y=y[:,i],alphas=self.alphas[i]['model'][0],positive=True)
            self.weight_.append(weight)
            self.model_.append(model)
        return self

    def predict(
        self, X: np.ndarray, alpha: list[np.ndarray] | None = None
    ) -> list[np.ndarray]:
        # pylint: disable=invalid-name
        """
        Make predictions

        Parameters
        ----------
        X : ndarray of shape (n_samples, p_features) or (n_samples, p_features, q_targets)
            Common feature matrix for all targets,
            or a specificfeature matrix for each target.
        alpha : List of length q_targets or None, default=None
            one ndarray of non-negative regularisation parameters for each target;
            if None, predictions are returned for the fitted path.

        Returns
        -------
        y_hat : list of length q_targets
            One ndarray of shape (n_samples, n_alphas) for each target.
        """
        check_is_fitted(self, attributes=["model_"])
        check_array(X, allow_nd=True, dtype="numeric")
        y_hat = []
        newxx = None
        if X.ndim == 2:
            newxx = np.hstack([X, -X])
        for i, _ in enumerate(self.model_):
            if X.ndim == 3:
                newxx = np.hstack([X[:, :, i], -X[:, :, i]])
            newx_scale = newxx * self.weight_[i]
            beta = self.model_[i][1]
            if alpha is None:
                y_hat.append((newx_scale @ beta) * self.sd_y_[i] + self.mu_y_[i])
            else:
                alpha_path = np.log(self.model_[i][0] + self._EPS)
                order_path = np.argsort(alpha_path)
                alpha_path = alpha_path[order_path]
                beta = beta[:, order_path]
                alpha_full = np.log(alpha[i] + self._EPS)
                # This avoids extrapolation outside range of fitted path:
                alpha_full = np.clip(alpha_full, alpha_path.min(), alpha_path.max())
                increase = np.hstack([True, np.diff(alpha_path) != 0])
                func_inter = interp1d(alpha_path[increase], beta[:, increase], axis=1)
                beta_inter = func_inter(alpha_full)
                y_hat.append((newx_scale @ beta_inter) * self.sd_y_[i] + self.mu_y_[i])
        return y_hat


class CoopLassoCV(RegressorMixin, BaseEstimator):
    # pylint: disable=too-many-instance-attributes
    """
    Cross-Validated Cooperative Multi-Task Lasso Regression.

    Fits cooperative multi-task lasso regression,
    optimising the regularisation parameters by cross-validation.

    Parameters
    ----------
    cv : int, default=10
        Number of cross-validation folds.
    n_alphas : int, default=100
        Number of candidate values for the regularisation parameter
        in the final regressions.
    l1_ratio : float, default=0.5
        Elastic net mixing parameter for the initial regressions,
        with `0<=l1_ratio<=1`,
        where `l1_ratio=0` leads to L2 (ridge)
        and `l1_ratio=1` leads to L1 (lasso) penalisation.
    exp_y : float, default=1.0
        Non-negative number for exponentiating
        the target-target correlation coefficients.
    exp_x : float, default=1.0
        Non-negative number for exponentiating
        the feature-feature correlation coefficients.
    random_state : int or None, default=None
        Random seed for generating reproducible cross-validation folds.

    Attributes
    ----------
    n_ : int
        Number of training samples.
    p_ : int
        Number of features.
    q_ : int
        Number of targets.
    model_ : list of length q_targets
        Fitted models from ``CoopLasso``.
    alpha_ : list length q_targets of ndarrays
        Sequence of regularisation parameters.
    mse_ : list of length q_targets of ndarrays
        Cross-validated mean squared errors
        for each value of alpha.
    min_ : list of length q_targets of ndarrays
        Indices of regularisation parameters corresponding to the lowest mean squared error.
    coef_ : ndarray of shape (q_targets, p_features)
        Estimated effects (of the feature in the column on the target in the row).

    See Also
    --------
    IndepLassoCV
        A convenience class using the same interface as ``CoopLassoCV``
        (similarly formatted inputs and outputs)
        without sharing information among targets or features.
    CoopLasso
        A class without cross-validation returning the lasso solution path.
        This is repeatedly called by ``CoopLassoCV``
        (once in each cross-validation iteration and once for the full dataset).

    Examples
    --------
    >>> from sklearn.datasets import load_linnerud
    >>> from collasso import CoopLassoCV
    >>> x, y = load_linnerud(return_X_y=True)
    >>> model = CoopLassoCV()
    >>> model.fit(x, y) # n_samples x p_features, n_samples x q_targets
    >>> model.coef_ # q_targets x p_features
    >>> y_pred = model.predict(x) # n_samples x q_targets
    """

    def __init__(
        self, *,
        cv: int = 10,
        n_alphas: int= 100,
        l1_ratio: float = 0.5,
        exp_y: float = 1,
        exp_x: float = 1,
        random_state: int|None = None
    ):
        # pylint: disable=too-many-arguments
        self.cv = cv
        self.n_alphas = n_alphas
        self.l1_ratio = l1_ratio
        self.exp_y = exp_y
        self.exp_x = exp_x
        self.random_state = random_state
        self.n_: int
        self.p_: int
        self.q_: int
        self.n_features_in_: int
        self.alpha_: list
        self.mse_: list
        self.min_: list
        self.model_: CoopLasso
        self.coef_: np.ndarray
        self.z_: np.ndarray|None

    def fit(
        self, X: np.ndarray, y: np.ndarray, Z: np.ndarray | None = None
    ) -> "CoopLassoCV":
        # pylint: disable=invalid-name
        """
        Fit CoopLassoCV

        Parameters
        ----------
        X : ndarray of shape (n_samples, p_features) or (n_samples, p_features, q_targets)
            common feature matrix for all targets or a specific feature matrix for each target
        y : ndarray of shape (n_samples, q_targets)
            target matrix
        Z : ndarray of shape (p_features,) or (p_features, q_targets), or None
            logical vector or matrix
            indicating primary (1, True) and auxiliary features (0, False)
            for all targets or each target

        Returns
        -------
        self : CoopLassoCV
            fitted model
        """
        X, y = _validate_train_data(self=self, X=X, y=y)
        self.n_, self.p_, self.q_ = _check_dims(X=X, y=y, Z=Z)
        self.n_features_in_ = self.p_
        self.z_ = Z
        self.model_ = CoopLasso(
            l1_ratio=self.l1_ratio,
            n_alphas=self.n_alphas,
            alpha_init=None,
            exp_y=self.exp_y,
            exp_x=self.exp_x,
        )
        self.model_.fit(X=X, y=y, Z=Z)
        self.alpha_ = []
        for i in range(self.q_):
            self.alpha_.append(self.model_.model_[i][0])
        y_hat = np.full((self.n_, self.q_, self.n_alphas), np.nan)
        folds = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        for train_id, test_id in folds.split(X=X, y=y):
            sub = CoopLasso(
                n_alphas=self.n_alphas,
                l1_ratio=self.l1_ratio,
                alpha_init=self.model_.alpha_init_,
                exp_y=self.exp_y,
                exp_x=self.exp_x,
            )
            sub.fit(X=X[train_id, ...], y=y[train_id, :], Z=Z)
            temp = sub.predict(X=X[test_id, ...], alpha=self.alpha_)
            for j, _ in enumerate(temp):
                y_hat[test_id, j, :] = temp[j]
        self.mse_ = []
        self.min_ = []
        self.coef_ = np.full((self.q_, self.p_), np.nan)
        for j in range(self.q_):
            mse = np.mean((y_hat[:, j, :] - y[:, j, np.newaxis]) ** 2, axis=0)
            self.mse_.append(mse)
            id_min = np.argmin(mse)
            self.min_.append(id_min)
            temp = self.model_.model_[j][1][:, id_min] * self.model_.weight_[j]
            self.coef_[j, :] = (
                temp[0 : self.p_] - temp[self.p_ : 2 * self.p_]
            ) * self.model_.sd_y_[j]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # pylint: disable=invalid-name
        """
        Make predictions

        Parameters
        ----------
        X : ndarray of shape (n_samples, p_features) or (n_samples, p_features, q_targets)
            Common feature matrix for all targets, or a specific feature matrix for each target.

        Returns
        -------
        y_hat : ndarray of shape (n_samples, q_targets)
            Matrix of predicted values (of the target in the column for the sample in the row).
        """
        check_is_fitted(self, attributes=["coef_"])
        X = _validate_test_data(self=self, X=X)
        y_hat = np.full((X.shape[0], self.q_), np.nan)
        newxx = None
        if X.ndim == 2:
            newxx = np.hstack([X, -X])
        for i in range(self.q_):
            if X.ndim == 3:
                newxx = np.hstack([X[:, :, i], -X[:, :, i]])
            newx_scale = newxx * self.model_.weight_[i]
            id_min = self.min_[i]
            beta = self.model_.model_[i][1][:, id_min]
            # fmt: off
            y_hat[:, i] = (newx_scale @ beta) * self.model_.sd_y_[i] + self.model_.mu_y_[i]
            # fmt: on
        if self.q_ == 1:
            return y_hat.ravel()
        return y_hat
