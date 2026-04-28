"""
Unit tests

Tests:
    test_broadcasting: matrix/array during training/testing
    test_interpolation: original/interpolated values
    test_compatibility: sklearn tests
"""

import numpy as np
import pytest
from sklearn.linear_model import LassoCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils.estimator_checks import parametrize_with_checks
from scipy.stats import spearmanr
from collasso import (
    CoopLassoCV,
    IndepLassoCV,
    simulate,
    _CoopLasso,
    _spearmanr,
    _calc_weights,
)


@pytest.fixture
def data():
    """Simulating data for unit tests"""
    x_train0, y_train0, x_test0, y_test0, beta0 = simulate(kappa=1)
    x_train1, y_train1, x_test1, y_test1, beta1 = simulate(kappa=0.5)
    assert beta0.shape == beta1.shape, "coef has same shape"
    assert y_train0.shape == y_train1.shape, "y_train has same shape"
    assert y_test0.shape == y_test1.shape, "y_test has same shape"
    assert x_train0.shape == x_train1[:, :, 0].shape, "x_train is one slice"
    assert x_test0.shape == x_test1[:, :, 0].shape, "x_test is one slice"
    return x_train0, y_train0, x_test0, y_test0, beta0


def test_wrapper(data):
    # pylint: disable=redefined-outer-name
    """Equivalence given common feature matrix"""
    x_train, y_train, x_test, _, _ = data
    model = IndepLassoCV(alphas=100, cv=5)
    model.fit(X=x_train, y=y_train)
    coef0 = model.coef_
    pred0 = model.predict(X=x_test)
    model = MultiOutputRegressor(LassoCV(alphas=100, cv=5))
    model.fit(X=x_train, y=y_train)
    coef1 = np.array([est.coef_ for est in model.estimators_])
    pred1 = model.predict(X=x_test)
    assert np.allclose(coef0, coef1), "coefficients should be the same"
    assert np.allclose(pred0, pred1), "predictions should be the same"


def test_singletask(data):
    # pylint: disable=redefined-outer-name
    """Single-task learning using multi-task learner"""
    x_train, y_train, x_test, _, _ = data
    y_train = np.column_stack((y_train[:, 0], y_train[:, 0]))
    # Explore exp=0 (duplicate problem) and exp=np.inf (problem with EPS)!
    model = CoopLassoCV(exp_y=0, exp_x=0, random_state=1)
    model.fit(X=x_train, y=y_train)
    coef0 = model.coef_
    pred0 = model.predict(X=x_test)
    assert np.allclose(coef0[0, :], coef0[1, :]), "same coefficients"
    assert np.allclose(pred0[:, 0], pred0[:, 1]), "same predictions"
    model.fit(X=x_train, y=y_train[:, 0])
    coef1 = model.coef_
    pred1 = model.predict(X=x_test)
    assert np.allclose(coef0[0, :], coef1), "same coefficients"
    assert np.allclose(pred0[:, 0], pred1), "same predictions"


def test_reconstruct_preds(data):
    # pylint: disable=redefined-outer-name
    """Reconstruct predictions from coefficients"""
    x_train, y_train, x_test, _, _ = data
    model = CoopLassoCV()
    model.fit(X=x_train, y=y_train)
    coef = model.coef_
    pred0 = model.predict(X=x_test)
    pred1 = x_test @ coef.T + model.model_.mu_y_
    assert np.allclose(pred0, pred1), "same predictions"


def test_broadcasting(data):
    # pylint: disable=redefined-outer-name
    """CoopLassoCV can use matrix or array during training or testing."""
    x_train, y_train, x_test, y_test, _ = data
    x_train_bc = np.broadcast_to(
        x_train[:, :, None], (x_train.shape[0], x_train.shape[1], y_train.shape[1])
    )
    x_test_bc = np.broadcast_to(
        x_test[:, :, None], (x_test.shape[0], x_test.shape[1], y_test.shape[1])
    )
    model = CoopLassoCV(random_state=1)
    model.fit(x_train, y_train)
    y_hat = []
    y_hat.append(model.predict(x_test))
    y_hat.append(model.predict(x_test_bc))
    model.fit(x_train_bc, y_train)
    y_hat.append(model.predict(x_test))
    y_hat.append(model.predict(x_test_bc))
    y_hat_array = np.array(y_hat)
    assert np.allclose(y_hat_array, y_hat_array[0]), "predictions should be the same"


def test_interpolation(data):
    # pylint: disable=redefined-outer-name
    """CoopLasso can use original or interpolated alpha values."""
    x_train, y_train, x_test, _, _ = data
    model = _CoopLasso()
    model.fit(X=x_train, y=y_train)
    pred1 = model.predict(X=x_test)
    alpha = []
    for i, _ in enumerate(model.model_):
        alpha.append(model.model_[i][0])
    pred2 = model.predict(X=x_test, alpha=alpha)
    assert np.allclose(pred1, pred2), "prediction should be the same"


def test_privileged_information(data):
    # pylint: disable=redefined-outer-name
    """CoopLassoCV estimates no coefficients for auxiliary variables"""
    x_train, y_train, x_test, _, _ = data
    p = x_train.shape[1]
    q = y_train.shape[1]
    z = np.random.binomial(n=1, p=0.5, size=p)
    model = CoopLassoCV(random_state=1)
    model.fit(X=x_train, y=y_train, Z=z)
    beta_hat = model.coef_[:, z == 0]
    y_hat0 = model.predict(X=x_test)
    xx_test = x_test.copy()
    xx_test[:,z==0] = np.nan
    y_hat1 = model.predict(X=xx_test)
    assert np.count_nonzero(beta_hat) == 0, "LUPI works under matrix y and vector z"
    assert np.allclose(y_hat0,y_hat1), "LUPI works under matrix y and vector z"
    model.fit(X=x_train, y=y_train[:, 0], Z=z)
    beta_hat = model.coef_[:, z == 0]
    y_hat0 = model.predict(X=x_test)
    y_hat1 = model.predict(X=xx_test)
    assert np.count_nonzero(beta_hat) == 0, "LUPI works under vectors y and z"
    assert np.allclose(y_hat0,y_hat1), "LUPI works under vectors y and z"
    z = np.random.binomial(n=1, p=0.5, size=(p, q))
    model.fit(X=x_train, y=y_train, Z=z)
    beta_hat = model.coef_[z.T == 0]
    y_hat0 = model.predict(X=x_test)
    xx_test = x_test.copy()
    xx_test[:, z.sum(axis=1) == 0] = np.nan
    y_hat1 = model.predict(X=xx_test)
    assert np.count_nonzero(beta_hat) == 0, "LUPI works under matrices y and z"
    assert np.allclose(y_hat0,y_hat1), "LUPI works under matrices y and z"


def test_weights(data):
    # pylint: disable=redefined-outer-name,too-many-locals
    """Weight calculation with loop or vector is the same"""
    x_train, y_train, _, _, coef = data
    exp_y = 1
    exp_x = 1
    p = x_train.shape[1]
    q = y_train.shape[1]
    cor_y = spearmanr(y_train).statistic
    cor_x = spearmanr(x_train).statistic
    # matrix notation (efficient)
    w_pos0 = np.full((p,q),np.nan)
    w_neg0 = np.full((p,q),np.nan)
    w_abs0 = np.full((p,q),np.nan)
    for i in range(q):
        w_pos, w_neg, w_abs = _calc_weights(
            cor_y=cor_y[:, i],
            cor_x=cor_x,
            coef=coef,
            exp_y=exp_y,
            exp_x=exp_x,
        )
        w_pos0[:, i] = w_pos
        w_neg0[:, i] = w_neg
        w_abs0[:, i] = w_abs
    # sigma notation (interpretable)
    w_pos1 = np.zeros((p,q))
    w_neg1 = np.zeros((p,q))
    w_abs1 = np.zeros((p,q))
    for j in range(p):
        for k in range(q):
            for j_ in range(p):
                for k_ in range(q):
                    cont = cor_x[j,j_]**exp_x*cor_y[k,k_]**exp_y*coef[j_,k_]
                    w_pos1[j,k] = w_pos1[j,k] + np.maximum(0,cont)
                    w_neg1[j,k] = w_neg1[j,k] + np.maximum(0,-cont)
                    w_abs1[j,k] = w_abs1[j,k] + np.abs(cont)
    assert np.allclose(w_pos0, w_pos1), "positive weights should be the same"
    assert np.allclose(w_neg0, w_neg1), "negative weights should be the same"
    assert np.allclose(w_abs0, w_abs1), "absolute weights should be the same"
    assert np.allclose(w_pos0 + w_neg0, w_abs0), "positive+negative=absolute"
    assert np.allclose(w_pos1 + w_neg1, w_abs1), "positive+negative=absolute"


def test_cor(data):
    # pylint: disable=redefined-outer-name
    """reproduce spearman cor"""
    x_train, _, _, _, _ = data
    cor1 = spearmanr(x_train).statistic
    cor2 = _spearmanr(x_train)
    assert np.allclose(cor1, cor2), "identical results"


@parametrize_with_checks([CoopLassoCV(), IndepLassoCV()])
def test_compatibility(estimator, check):
    """compatibility with scikit-learn"""
    check(estimator)


#  This requires examples in docstrings:
#  import doctest
#  def test_docstrings_collasso():
#     file = "C:/Users/arauschenberger/Desktop/collasso/src/collasso/functions.py"
#     results = doctest.testfile(file,module_relative=False)
#     assert results.failed == 0
