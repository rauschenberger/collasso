"""
Simulation

This script performs a simulation study to compare the selection performance
and the predictive performance between different methods.

"""

from itertools import product
import numpy as np
from scipy.stats import  ttest_rel
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from collasso import simulate, SingleTaskLassoCV, CoopLassoCV

# Consider decreasing q from 10 to 3 (faster and more realistic).

rho = [0, 0.90]
prob_com = [0.00, 0.05]
prob_sep = [0.00, 0.05]
n_iter = range(10)
grid = np.array(list(product(rho, prob_com, prob_sep, n_iter)))
grid = grid[(grid[:,1]!=0) | (grid[:,2]!=0),:] # remove if prob_com=prob_sep=0

propnzero = np.full((len(grid),4),np.nan)
precision = np.full((len(grid),4),np.nan)
prederror = np.full((len(grid),4),np.nan)

for i,_ in enumerate(grid):
    np.random.seed(i)
    # Try with common feature matrix (kappa=1)
    # and with separate feature matrices (0<=kappa<1).
    kappa = 0.5
    x_train, y_train, x_test, y_test, beta = simulate(
        rho=grid[i,0],
        prob_com=grid[i,1],
        prob_sep=grid[i,2],
        kappa = kappa
    )
    if kappa ==1:
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
    else:
        x_train_scaled = x_train*np.nan
        for k in range(x_train.shape[2]):
            scaler = StandardScaler()
            x_train_scaled[:,:,k] = scaler.fit_transform(x_train[:,:,k])
            x_test_scaled[:,:,k] = scaler.transform(x_test[:,:,k])
    # prediction by the mean
    #coef_mean = np.zeros(beta.T.shape)
    coef_mean = np.zeros((y_train.shape[1],x_train.shape[1]))
    pred_mean = np.tile(np.mean(y_train,axis=0),(y_test.shape[0],1))
    # separate lasso regressions (also if separate X)
    model_single = SingleTaskLassoCV(alphas=100,cv=5)
    model_single.fit(X=x_train_scaled,y=y_train)
    coef_single = model_single.coef_
    pred_single = model_single.predict(X=x_test_scaled)
    if kappa ==1:
        # multi-task lasso regression
        model_multi = MultiTaskLassoCV(alphas=100,cv=5)
        model_multi.fit(X=x_train_scaled,y=y_train)
        coef_multi = model_multi.coef_
        pred_multi = model_multi.predict(X=x_test_scaled)
    else:
        coef_multi = np.full((y_train.shape[1],x_train.shape[1]),np.nan)
        pred_multi = np.full((y_test.shape[0],y_test.shape[1]),np.nan)
    # cooperative lasso regression
    model_coop = CoopLassoCV(n_alphas=100,cv=5,l1_ratio=0.5)
    model_coop.fit(X=x_train_scaled,y=y_train)
    coef_coop = model_coop.coef_
    pred_coop = model_coop.predict(X=x_test_scaled)

    # mutar (add internal cross-validation)
    #x_train_scaled_repeated = np.repeat(x_train_scaled[None,:,:],repeats=y_train.shape[1],axis=0)
    #x_test_scaled_repeated = np.repeat(x_test_scaled[None,:,:],repeats=y_test.shape[1],axis=0)
    #
    #model = GroupLasso()
    #model.fit(x_train_scaled_repeated,y_train.T)
    #coef_group = model.coef_.T
    #pred_group = model.predict(x_test_scaled_repeated).T
    #
    #model = DirtyModel()
    #model.fit(x_train_scaled_repeated,y_train.T)
    #coef_dirty = model.coef_.T
    #pred_dirty = model.predict(x_test_scaled_repeated).T
    #
    #model = MultiLevelLasso()
    #model.fit(x_train_scaled_repeated,y_train.T)
    #coef_level = model.coef_.T
    #pred_level = model.predict(x_test_scaled_repeated).T

    # comparison
    if kappa ==1:
        coef = np.stack([coef_mean,coef_single,coef_multi,coef_coop])
        pred = np.stack([pred_mean,pred_single,pred_multi,pred_coop])
    else:
        coef = np.stack([coef_mean,coef_single,coef_coop])
        pred = np.stack([pred_mean,pred_single,pred_coop])
    for j,_ in enumerate(coef):
        propnzero[i,j] = np.mean(coef[j]!=0)
        if np.sum(coef[j])==0:
            precision[i,j] = np.nan
        else:
            precision[i,j] = np.sum((beta.T!=0) & (coef[j]!=0))/np.sum(coef[j]!=0)
        prederror[i,j] = mean_squared_error(y_test,pred[j],multioutput='uniform_average')

# summary
np.mean(propnzero,axis=0)
np.mean(precision,axis=0)
predratio = (prederror.T/prederror[:,0]).T
np.mean(predratio,axis=0)
#array([1.        , 0.31545336, 0.32678457, 0.30478868])

ttest_rel(a=predratio[:,1],b=predratio[:,2])
ttest_rel(a=predratio[:,1],b=predratio[:,3])
ttest_rel(a=predratio[:,2],b=predratio[:,3])

predratio.reshape(6,10,-1).mean(axis=1)
