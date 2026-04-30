"""
Vignette.

This script illustrates sparse linear multi-task regression with
(i) a common feature matrix, (ii) specific feature matrices,
and (iii) privileged information.
"""

# %% [markdown]
# # Examples
#
# Use the class `CoopLassoCV` for multi-task regression
# (sharing information between targets and features).
# For comparison, use the class `IndepLassoCV` for
# independent lasso regressions for multiple targets.

# %% [markdown]
# ## Initialisation
# Import the function `simulate` to simulate data,
# and import the class `CoopLassoCV` to perform linear multi-task regression.

# %%
import numpy as np
from sklearn.metrics import mean_squared_error, precision_score
from collasso import simulate, CoopLassoCV

# %% [markdown]
# ## (i) Multi-task regression with a common feature matrix
#
# The standard setting for multi-task regression involves
# a feature matrix of shape `(n_samples, p_features)`
# and a target matrix of shape `(n_samples, q_targets)`.
#
# Model training requires the feature matrix
# and the target matrix of the training samples (`x_train` and `y_train`),
# model testing requires the feature matrix of the testing samples (`x_test`).

# %% [markdown]
# Simulate training and test data:

# %%
x_train, y_train, x_test, y_test, beta = simulate()

# %% [markdown]
# Fit linear multi-task regression:

# %%
model = CoopLassoCV()
model.fit(X=x_train, y=y_train)

# %% [markdown]
# Extract estimated coefficients, a matrix of shape `(p_features, q_targets)`,
# and calculate the precision:

# %%
beta_hat = model.coef_.T  # estimated regression coefficients
precision_score(y_true=beta!=0, y_pred=model.coef_.T!=0, average="micro")

# %% [markdown]
# Make out-of-sample predictions, a matrix of shape (`n_samples, q_targets`),
# and calculate the mean squared error:

# %%
y_hat = model.predict(X=x_test)
mean_squared_error(y_true=y_test, y_pred=y_hat)

# %% [markdown]
# ## (ii) Multi-task regression with specific feature matrices
#
# In some settings, there is not a common feature matrix for all targets
# but a specific feature matrix for each target.
# Then the model requires a feature array of shape `(n_samples, p_features, q_targets)`.

# %% [markdown]
# Simulate training and test data:

# %%
x_train, y_train, x_test, y_test, beta = simulate(kappa=0.5)

# %% [markdown]
# Fit linear multi-task regression:

# %%
model = CoopLassoCV()
model.fit(X=x_train, y=y_train)

# %% [markdown]
# Extract estimated coefficients, a matrix of shape `(p_features, q_targets)`,
# and calculate the precision:

# %%
beta_hat = model.coef_.T
precision_score(y_true=beta!=0, y_pred=model.coef_.T!=0, average="micro")

# %% [markdown]
# Make out-of-sample predictions, a matrix of shape (`n_samples, q_targets`),
# and calculate the mean squared error:

# %%
y_hat = model.predict(X=x_test)
mean_squared_error(y_true=y_test, y_pred=y_hat)

# %% [markdown]
# ## (iii) Multi-task regression with privileged information
#
# In some applications, some features may be used for model training
# but not for model testing (prileged information).
# In contrast to primary features,
# auxiliary features must not be selected by the model.
# Irrespective of whether there is a common feature matrix for all targets
# or a specific feature matrix for each target,
# it is possible to exclude the same set of features for all targets,
# or specific sets of features for each target.

# %% [markdown]
# **Step 1** - Simulate data

# %% [markdown]
# **Option A**: common feature matrix

# %%
x_train, y_train, x_test, y_test, beta = simulate()

# %% [markdown]
# **Option B**: separate feature matrices

# %%
x_train, y_train, x_test, y_test, beta = simulate(kappa=0.5)

# %% [markdown]
# **Step 2** - Define primary and auxiliary features

# %% [markdown]
# **Option A**:
# Allow the model to select from the *same set* of features for all targets,
# by defining `z` as a vector of shape `(p_features, )`:

# %%
z = np.zeros(x_train.shape[1])
z[0:100] = 1

# %% [markdown]
# (Here, all targets have the primary features `x_1,...,x_100`
# and the auxiliary features `x_101,...,x_200`.)

# %% [markdown]
# **Option B**:
# Allow the model to select from a *different set* of features for each target,
# by defining `z` as a matrix of shape `(p_features, q_targets`):

# %%
z = np.zeros((x_train.shape[1],y_train.shape[1]))
z[0:50,0] = 1
z[100:175,1] = 1
z[125:200,2] = 1

# %% [markdown]
# (Here, the first target has the primary features `x_1,...,x_50`,
# the second target `x_101,...,x_175`,
# and the third target `x_126,...,x_200`.)

# %% [markdown]
# Fit linear multi-task regression:

# %%
model = CoopLassoCV()
model.fit(X=x_train, y=y_train, Z=z)

# %% [markdown]
# Extract estimated coefficients, a matrix of shape `(p_features, q_targets)`,
# and calculate precision:

# %%
y_hat = model.predict(X=x_test)
beta_hat = model.coef_.T

# %% [markdown]
# **Note**: As auxiliary features are not selected,
# their values in the test data have no impact on predictions.

# %%
assert np.all(beta_hat[z == 0, ...] == 0)
x_test_new = x_test.copy()
if x_test.ndim == 2 and z.ndim == 2:
    x_test_new[: , np.sum(z,axis=1) == 0] = np.nan
else:
    x_test_new[:, z == 0, ...] =  np.nan
assert np.all(y_hat == model.predict(x_test_new))

# %% [markdown]
# ## Related packages
#
# - **scikit-learn** implements
# linear multi-task lasso and elastic net regression,
# in the classes `MultiTaskLasso`, `MultiTaskLassoCV`,
# `MultiTaskElasticNet` and `MultiTaskElasticNetCV`.<br>
# [GitHub](https://github.com/scikit-learn/scikit-learn)|
# [PyPI](https://pypi.org/project/scikit-learn/)|
# [website](https://scikit-learn.org)
#
# - **MuTaR** from Hicham Janati implements
# group-norms multi-task linear models
# and optimal transport regularised models.<br>
# [GitHub](https://github.com/hichamjanati/mutar)|
# [PyPI](https://pypi.org/project/mutar/)|
# [website](https://hichamjanati.github.io/mutar/)
#
# - **scikit-MTR** from Henzhe Zhang implements
# multi-task regression by stacking. <br>
# [GitHub](https://github.com/hengzhe-zhang/Scikit-MTR)|
# [PyPI](https://pypi.org/project/scikit-MTR/)
