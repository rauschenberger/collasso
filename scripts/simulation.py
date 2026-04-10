"""Simulation

This script performs a simulation study to compare the selection performance and the predictive performance between different methods.

"""

from itertools import product
import numpy as np
from scipy.stats import  ttest_rel
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from collasso import simulate, SingleTaskLassoCV, CoopLassoCV

rho = [0, 0.90]
prob_com = [0.00, 0.05]
prob_sep = [0.00, 0.05]
n_iter = range(10)
grid = np.array(list(product(rho, prob_com, prob_sep, n_iter)))
grid = grid[(grid[:,1]!=0) | (grid[:,2]!=0),:] # remove if prob_com=prob_sep=0

propnzero = np.full((len(grid),4),np.nan)
precision = np.full((len(grid),4),np.nan)
prederror = np.full((len(grid),4),np.nan)

for i in range(len(grid)):
    np.random.seed(i)
    x_train, y_train, x_test, y_test, beta = simulate(rho=grid[i,0], prob_com=grid[i,1], prob_sep=grid[i,2])
    #X, y = x_train, y_train # temporary
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_train_scaled_repeated = np.repeat(x_train_scaled[None,:,:],repeats=y_train.shape[1],axis=0)
    x_test_scaled = scaler.transform(x_test)
    x_test_scaled_repeated = np.repeat(x_test_scaled[None,:,:],repeats=y_test.shape[1],axis=0)
    # prediction by the mean
    coef_mean = np.zeros(beta.T.shape)
    pred_mean = np.tile(np.mean(y_train,axis=0),(y_test.shape[0],1))
    # separate lasso regressions
    model = SingleTaskLassoCV(alphas=100,cv=5)
    model.fit(X=x_train_scaled,y=y_train)
    coef_single = model.coef_
    pred_single = model.predict(X=x_test_scaled)
    # multi-task lasso regression
    model = MultiTaskLassoCV(alphas=100,cv=5)
    model.fit(X=x_train_scaled,y=y_train)
    coef_multi = model.coef_
    pred_multi = model.predict(X=x_test_scaled)
    # cooperative lasso regression
    model = CoopLassoCV(n_alphas=100,cv=5,l1_ratio=0.5)
    model.fit(X=x_train_scaled,y=y_train)
    coef_coop = model.coef_
    pred_coop = model.predict(X=x_test_scaled)
    
    # mutar (add internal cross-validation)
    
    #model = GroupLasso()
    #model.fit(x_train_scaled_repeated,y_train.T)
    #coef_group = model.coef_.T
    #pred_group = model.predict(x_test_scaled_repeated).T
  
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
    coef = np.stack([coef_mean,coef_single,coef_multi,coef_coop])
    pred = np.stack([pred_mean,pred_single,pred_multi,pred_coop])
    for j in range(len(coef)):
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
