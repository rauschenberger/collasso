
import numpy as np
import pytest
from collasso import simulate, CoopLasso, CoopLassoCV

@pytest.fixture
def data():
    x_train, y_train, x_test, y_test, beta = simulate(rho=0.9, prob_com=0.05, prob_sep=0.05)

def test_matrix_array_equivalence():
    """CoopLassoCV can use matrix or array during training or testing."""
    x_train_bc = np.broadcast_to(x_train[:,:,None],(x_train.shape[0],x_train.shape[1],y_train.shape[1]))
    x_test_bc = np.broadcast_to(x_test[:,:,None],(x_test.shape[0],x_test.shape[1],y_test.shape[1]))
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

# test whether CoopLasso returns same results if alpha=None and alpha is provided from fitted path

def test_path_interpolate_equivalence():
    """CoopLasso can use original or interpolated alpha values."""
    model = CoopLasso()
    model.fit(X=x_train,y=y_train)
    pred1 = model.predict(X=x_test)
    alpha = []
    for i in range(len(model.model_)):
        alpha.append(model.model_[i][0])
    pred2 = model.predict(X=x_test,alpha=alpha)
    assert np.allclose(pred1,pred2), 'prediction should be the same'
