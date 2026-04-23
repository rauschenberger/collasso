
# --- initialisation ---

import numpy as np
from sklearn.metrics import mean_squared_error, precision_score
from collasso import simulate, CoopLassoCV  # alternative: IndepLassoCV
model = CoopLassoCV()  # alternative: model = IndepLassoCV()


# --- multi-task regression with a common feature matrix ---

x_train, y_train, x_test, y_test, beta = simulate()
# y_train.shape  # features (n x p matrix)
# x_train.shape  # targets (n x q matrix)

model.fit(X=x_train, y=y_train)

y_hat = model.predict(X=x_test) # out-of-sample predicted values (n x q matrix)
beta_hat = model.coef_.T  # estimated regression coefficients (p x q matrix)

mean_squared_error(y_true=y_test, y_pred=y_hat)
precision_score(y_true=beta!=0, y_pred=model.coef_.T!=0, average="micro")


# --- multi-task regression with specific feature matrices ---

x_train, y_train, x_test, y_test, beta = simulate(kappa=0.5)
# x_train.shape  # features (n x p x q array)
# y_train.shape

model.fit(X=x_train, y=y_train)

y_hat = model.predict(X=x_test)
beta_hat = model.coef_.T

mean_squared_error(y_true=y_test, y_pred=y_hat)
precision_score(y_true=beta!=0, y_pred=model.coef_.T!=0, average="micro")


# --- multi-task regression with privileged information ---

x_train, y_train, x_test, y_test, beta = simulate()
z = np.random.binomial(n=1, p=0.5, size=x_test.shape[1])

model.fit(X=x_train, y=y_train, Z=z)

y_hat = model.predict(X=x_test)
beta_hat = model.coef_.T

# Auxiliary features are not selected:
np.all(beta_hat[z == 0, :] == 0)  # no selection

# And their values in the test data therefore have no impact on predictions:
x_test_new = x_test
x_test_new[:, z == 0] =  np.nan
np.all(y_hat == model.predict(x_test_new))  # no impact
