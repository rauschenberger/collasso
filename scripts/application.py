"""
Application

This script compares the predictive performance between different methods in an application.
"""

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import openml
from collasso import SingleTaskLassoCV, CoopLassoCV

#from sklearn.datasets import load_linnerud # move to examples
#x, y = load_linnerud(return_X_y=True) # move to examples

# scm1d
dataset = openml.datasets.get_dataset(dataset_id=41485,version=2)
data = dataset.get_data()
assert isinstance(data[0], pd.DataFrame)
x = data[0].iloc[:, :-16].values
y = data[0].iloc[:, -16:].values

pred = np.full((y.shape[0],y.shape[1],4),np.nan)

np.random.seed(0)

folds = KFold(n_splits=2,shuffle=True,random_state=1)
for train_id, test_id in folds.split(X=x,y=y):
    y_train = y[train_id,:]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_id,:])
    x_test = scaler.transform(x[test_id,:])
    # prediction by the mean
    pred[test_id,:,0] = np.mean(y[train_id,:],axis=0)
    # single-task lasso regression
    model = SingleTaskLassoCV(alphas=100,cv=5)
    model.fit(X=x_train,y=y_train)
    pred[test_id,:,1] = model.predict(X=x_test)
    # multi-task lasso regression
    model = MultiTaskLassoCV(alphas=100,cv=5)
    model.fit(X=x_train,y=y_train)
    pred[test_id,:,2] = model.predict(X=x_test)
    # cooperative lasso regression
    model = CoopLassoCV(n_alphas=100,cv=5,l1_ratio=0.5)
    model.fit(X=x_train,y=y_train)
    pred[test_id,:,3] = model.predict(X=x_test)

mse = np.mean((pred[:,:,:] - y[:,:,np.newaxis])**2, axis=0)
#mean_squared_error(y,pred[:,:,3],multioutput='raw_values')

# mean proportion of mse w.r.t. to empty model
np.mean(mse/mse[:,[0]],axis=0)

# mean rank
ranks = np.apply_along_axis(rankdata,axis=1,arr=mse)
np.mean(ranks,axis=0)
