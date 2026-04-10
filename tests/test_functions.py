"""Unit tests

Tests:
    test_broadcasting: matrix/array during training/testing
    test_interpolation: original/interpolated values
    test_compatibility: sklearn tests
"""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks
from collasso import CoopLasso, CoopLassoCV, simulate, SingleTaskLassoCV

@pytest.fixture
def data():
    """Simulating data for unit tests"""
    x_train, y_train, x_test, y_test, beta = simulate(
        rho=0.9,
        prob_com=0.05,
        prob_sep=0.05
    )
    return x_train, y_train, x_test, y_test, beta

def test_broadcasting(data): # pylint: disable=redefined-outer-name
    """CoopLassoCV can use matrix or array during training or testing."""
    x_train, y_train, x_test, y_test, _ = data
    x_train_bc = np.broadcast_to(
        x_train[:,:,None],
        (x_train.shape[0],x_train.shape[1],y_train.shape[1])
    )
    x_test_bc = np.broadcast_to(
        x_test[:,:,None],
        (x_test.shape[0],x_test.shape[1],y_test.shape[1])
    )
    model = CoopLassoCV(random_state=1)
    model.fit(x_train,y_train)
    y_hat = []
    y_hat.append(model.predict(x_test))
    y_hat.append(model.predict(x_test_bc))
    model.fit(x_train_bc,y_train)
    y_hat.append(model.predict(x_test))
    y_hat.append(model.predict(x_test_bc))
    y_hat = np.array(y_hat)
    assert np.allclose(y_hat, y_hat[0]), 'predictions should be the same'

def test_interpolation(data): # pylint: disable=redefined-outer-name
    """CoopLasso can use original or interpolated alpha values."""
    x_train, y_train, x_test, _, _ = data
    model = CoopLasso()
    model.fit(X=x_train,y=y_train)
    pred1 = model.predict(X=x_test)
    alpha = []
    for i in range(len(model.model_)):
        alpha.append(model.model_[i][0])
    pred2 = model.predict(X=x_test,alpha=alpha)
    assert np.allclose(pred1,pred2), 'prediction should be the same'

#SKIP = {
#    "",
#    # ...
#}

#@parametrize_with_checks([CoopLassoCV(), SingleTaskLassoCV()])
#def test_compatibility(estimator, check):
#    if check.func.__name__ in SKIP:
#        pytest.skip("skipped")
#    check(estimator)

#import collasso
#
#def test_docstrings_collasso():
#    results = doctest.testmod(collasso, verbose=False)
#    assert results.failed == 0
