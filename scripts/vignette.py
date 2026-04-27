"""
Vignette.

This vignette illustrates sparse linear multi-task regression
with a common feature matrix, with specific feature matrices,
and with privileged information.

Use the class ``CoopLassoCV`` for multi-task regression
(sharing information between targets and features).

For comparison, use the class ``IndepLassoCV`` for
independent lasso regressions for multiple targets.
"""


# --- initialisation ---

import numpy as np
from sklearn.metrics import mean_squared_error, precision_score
from collasso import simulate, CoopLassoCV


# --- multi-task regression with a common feature matrix ---

# The standard setting for multi-task regression involves
# a feature matrix of shape (n_samples, p_features)
# and a target matrix of shape (n_samples, q_targets).

# Model training requires the feature matrix
# and the target matrix of the training samples (x_train and y_train),
# model testing requires the feature matrix of the testing samples (x_test).

x_train, y_train, x_test, y_test, beta = simulate()

model = CoopLassoCV()
model.fit(X=x_train, y=y_train)

beta_hat = model.coef_.T  # estimated regression coefficients (p x q matrix)
precision_score(y_true=beta!=0, y_pred=model.coef_.T!=0, average="micro")

y_hat = model.predict(X=x_test) # out-of-sample predicted values (n x q matrix)
mean_squared_error(y_true=y_test, y_pred=y_hat)



# --- multi-task regression with specific feature matrices ---

# In some settings, there is not a common feature matrix for all targets
# but a specific feature matrix for each target.
# Then the model requires a feature array of shape (n_samples, p_features, q_targets).

x_train, y_train, x_test, y_test, beta = simulate(kappa=0.5)

model = CoopLassoCV()
model.fit(X=x_train, y=y_train)

beta_hat = model.coef_.T
precision_score(y_true=beta!=0, y_pred=model.coef_.T!=0, average="micro")

y_hat = model.predict(X=x_test)
mean_squared_error(y_true=y_test, y_pred=y_hat)



# --- multi-task regression with privileged information ---

# In some applications, some features may be used for model training
# but not for model testing (prileged information).
# In contrast to primary features,
# auxiliary features must not be selected by the model.
# It is possible to exclude the same set of features for all targets,
# or specific sets of features for each target.

x_train, y_train, x_test, y_test, beta = simulate()

# Allow the model to select from the same set of features for all targets:
z = np.zeros(x_train.shape[1])
z[0:100] = 1
# Here, all targets have the same
# primary features (x_1,...,x_100)
# and auxiliary features (x_101,...,x_200).

# Allow the model to select from a different set of features for each target:
z = np.zeros((x_train.shape[1],y_train.shape[1]))
z[0:100,0] = 1
z[100:150,1] = 1
z[150:200,2] = 1

model = CoopLassoCV()
model.fit(X=x_train, y=y_train, Z=z)

y_hat = model.predict(X=x_test)
beta_hat = model.coef_.T

# Auxiliary features are not selected:
if z.ndim==1:  # z = vector of shape (p_features,)
    np.all(beta_hat[z == 0, :] == 0)
else:  # z = matrix shape (p_features, q_targets)
    np.all(beta_hat[z == 0] == 0)

# And their values in the test data therefore have no impact on predictions:
x_test_new = x_test
x_test_new[:, z == 0] =  np.nan
np.all(y_hat == model.predict(x_test_new))  # no impact



# --- related Python packages ---

# scikit-learn provides the classes MultiTaskLasso, MultiTaskLassoCV.
# MultiTaskElasticNet, MultiTaskElasticNetCV
# GitHub: https://github.com/scikit-learn/scikit-learn
# PyPI: https://pypi.org/project/scikit-learn/
# website: https://scikit-learn.org

# MuTaR from Hicham Janati:
# group-norms multi-task linear models and optimal transport regularised models
# GitHub: https://github.com/hichamjanati/mutar
# PyPI: https://pypi.org/project/mutar/
# website: https://hichamjanati.github.io/mutar/

# scikit-MTR from Henzhe Zhang:
# multi-task regression by stacking
# (interpretable feature-target effects can be obtained
# if linear regression is used in the base and in the meta layer)
# GitHub: https://github.com/hengzhe-zhang/Scikit-MTR
# PyPI: https://pypi.org/project/scikit-MTR/
