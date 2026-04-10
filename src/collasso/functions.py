"""Sparse linear multi-task regression 
with correlation-based information sharing

Classes:
    SingleTaskLassoCV: Modelling multiple targets separately
    CoopLassoCV: Modelling multiple targets together

Example:
    # from collasso import CoopLassoCV
    # x_train, y_train, x_test, y_test, beta = simulate()
    # model = CoopLassoCV()
    # model.fit(x_train, y_train)
    # model.predict(x_test)
"""

import numpy as np
from scipy.interpolate import interp1d # switch to np.interp
from scipy.stats import multivariate_normal, rankdata, spearmanr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet, ElasticNetCV, lasso_path, LassoCV
from sklearn.model_selection import KFold
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

#--- simulate data ---

def simulate(
    n0=100,
    n1=10000,
    p=200,
    q=10,
    rho=0.90,
    prob_com=0.05,
    prob_sep=0.05,
    common=True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate Data for Linear Multi-Task Regression
    
    Simulates feature matrix and target matrix,
    with given probabilities of
    (i) common effects on all targets and
    (ii) separate effects on each target.
    
    Parameters
    ----------
    
    n0 : int, default=100
        number of training samples
    n1 : int, default=10000
        number of testing samples
    p : int, default=200
        number of features
    q : int, default=10
        number of targets
    rho : float, default=0.90
        correlation coefficient, 0<=rho<=1
    prob_com : float, default=0.05
        probability of common effects for all targets, 0<=prob_com<=1
    prob_sep : float, default=0.05
        probability of separate effects for each target
      
    Returns
    -------
    x_train : ndarray of shape (n0_samples,p_features)
        training feature matrix
    y_train : ndarray of shape (n0_samples,q_targets)
        training target matrix
    x_test : ndarray of shape (n1_samples,p_features)
        test feature matrix
    y_test : ndarray of shape (n1_samples,q_targets)
        test target matrix
    beta : ndarray of shape (p_features,q_targets)
        true effects in the training and the test data
        (of the feature in the row on the target in the column)
        
    """
    # parameters
    n = n0 + n1
    fold = np.array([0]*n0+[1]*n1)
    # features
    mean = np.zeros(p)
    idx = np.arange(p)
    row_idx, col_idx = np.meshgrid(idx, idx)
    sigma = rho ** np.abs(col_idx - row_idx)
    if common is True:
        x = multivariate_normal.rvs(mean=mean, cov=sigma, size=n)
    else:
        x = []
        for k in range(q):
            x.append(multivariate_normal.rvs(mean=mean, cov=sigma, size=n))
    # effects
    beta_com = (
        np.random.binomial(n=1, p=prob_com, size=p) *
        np.abs(np.random.normal(size=p))
    )
    beta_sep = (
        np.random.binomial(n=1, p=prob_sep, size=p * q).reshape(p, q) *
        np.abs(np.random.normal(size=p * q)).reshape(p, q)
    )
    beta = beta_com[:, np.newaxis] + beta_sep
    # targets
    if common is True:
        eta = x @ beta
    else:
        eta = np.full((n,q),np.nan)
        for k in range(q):
            eta[:,k] = x[k] @ beta[:,k]
    noise_sd = 0.5 * np.std(eta, axis = 0)
    y = eta + np.random.normal(size = eta.shape, scale = noise_sd)
    if common is False:
        raise NotImplementedError("Returning multiple feature matrices is not implemented.")
    x_train, y_train = x[fold==0], y[fold==0]
    x_test, y_test = x[fold==1], y[fold==1]
    return x_train, y_train, x_test, y_test, beta

#--- single-task lasso regressions ---

class SingleTaskLassoCV(BaseEstimator,RegressorMixin):
    """
    Single-Task Lasso Regression For Multiple Targets
    
    Fits single-task lasso regression separately to multiple targets,
    optimising the regularisation parameters by cross-validation.

    Parameters
    ----------
    cv : int
        number of cross-validation folds
    alphas : int
        number of candidate values for the regularisation parameter
        
    Attributes
    ----------
    n_ : int
        number of training samples
    p_ : int
        number of features
    q_ : int
        number of targets
    model_ : list of length q_targets
        fitted models from LassoCV (one for each target)   
    coef_ : ndarray of shape (q_targets, p_features)
        estimated coefficients
        (of the feature in the column on the target in the row)
        
    Methods
    -------
    fit(X,y)
        Fits the models
    predict(X)
        Makes predictions
    """
    def __init__(self,cv=10,alphas=100):
        """
        cv : int, default=10
            number of cross-validation folds
        alphas : int, default=100
            number of candidate values for the regularisation parameter
        """
        self.cv = cv
        self.alphas = alphas
        self.n_ = self.p_ = self.q_ = self.model_ = self.coef_ = None
    def fit(self,X:np.ndarray,y:np.ndarray) -> "SingleTaskLassoCV": # pylint: disable=invalid-name
        """
        Fit SingleTaskLassoCV
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, p_features) or (n_samples, p_features, q_targets)
            common feature matrix for all targets or
            a separate feature matrix for each target
        y : ndarray of shape (n_samples, q_targets)
            target matrix
          
        Returns
        -------
        
        self: SingleTaskLassoCV
            fitted model
        """
        if y.ndim==1:
            y = y.reshape(-1,1)
        check_array(array=X,allow_nd=True)
        check_array(array=y)
        if X.ndim==2:
            X = np.broadcast_to(X[:, :, None], (X.shape[0], X.shape[1], y.shape[1]))
        self.n_, self.p_, self.q_ = _check_dims(X=X,y=y,Z=None)
        self.model_ = []
        self.coef_ = np.full((self.q_,self.p_), np.nan)
        for i in range(self.q_):
            model = LassoCV(alphas=self.alphas,cv=self.cv)
            model.fit(X[:, :, i],y[:, i])
            self.model_.append(model)
            self.coef_[i,:] = model.coef_
        return self
    def predict(self,X:np.ndarray) -> np.ndarray: # pylint: disable=invalid-name
        """
        Make predictions
  
        Parameters
        ----------
        X : ndarray of shape (n_samples, p_features) or (n_samples, p_features, q_targets)
            common feature matrix for all targets,
            or a separate feature matrix for each target
        
        Returns
        -------
        y_hat : ndarray of shape (n_samples, q_targets)
            matrix of predicted values
        """
        if X.ndim==2:
            X = np.broadcast_to(X[:, :, None], (X.shape[0], self.p_, self.q_))
        check_is_fitted(self,attributes=['coef_'])
        y_hat = np.full((X.shape[0],self.q_), np.nan)
        for i in range(self.q_):
            y_hat[:, i] = self.model_[i].predict(X[:,:,i])
        return y_hat

#--- multi-task lasso regression ---

def _check_dims(X:np.ndarray,y:np.ndarray,Z:np.ndarray|None): # pylint: disable=invalid-name
    """
    Check dimensionality of inputs
    
    Parameters
    ----------
        X np.ndarray
        y np.ndarray
        Z np.ndarray
    
    """
    #--- targets ---
    if y.ndim!=2:
        raise ValueError("'y' should be an 'n x q' matrix")
    n, q = y.shape

    #--- features ---
    if X.ndim not in (2,3):
        raise ValueError("'X' should be an 'n x p' matrix or an 'n x p x q' array")
    if X.shape[0]!=n:
        raise ValueError(
            "'y' and 'X' should have the same number of samples"
            "(first dimension in 'y' and 'X')"
        )
    if X.ndim==3 and X.shape[2]!=q:
        raise ValueError(
            "'y' and 'X' should have the same number of targets"
            "(second dimension in 'y', third dimension in 'X')"
        )
    p = X.shape[1]

    #--- indicators ---
    if Z is not None:
        if Z.ndim not in (1,2):
            raise ValueError("'Z' should be a 'p' vector or an 'p x q' matrix")
        if (Z.ndim==1 and Z.shape[0]!=p) or (Z.ndim==2 and Z.shape[0]!=p):
            raise ValueError(
                "'X' and 'Z' should have the same number of features"
                "(second dimension in 'X', first dimension in 'Z')"
            )
        if Z.ndim==2 and Z.shape[1]!=q:
            raise ValueError(
                "'y' and 'Z' should have the same number of targets"
                "(second dimension in 'y' and 'Z')"
            )
    return n, p, q

class CoopLasso(BaseEstimator,RegressorMixin):
    """
    Cooperative Multi-Task Lasso Regression
  
    Fits cooperative multi-task lasso regression

    Parameters
    ----------
    n_alphas : int, default=100
        number of candidate values for the regularisation parameter in the final regression
    l1_ratio : float, default=0.5
        elastic net mixing parameter for initial regression, ratio in [0,1]
    alpha_init : ndarray of shape (q_targets,) or None, default=None    
        regularisation parameters for the initial regressions, non-negative number
    exp_y : float, default=1.0
        exponent for target-target correlation coefficients, non-negative number
    exp_x : float, default=1.0
        exponent for feature-feature correlation coefficients, non-negative number
        
    Attributes
    ----------
    n_ : int
        number of training samples
    p_ : int
        number of features
    q_ : int
        number of targets
    model_ : list
        list of q_ models (one for each target)
        fitted by sklearn.linear_model.lasso_path
        with concatenated identity and inverse of feature matrix (X,-X)
        and non-negativity constraint (positive=True)
        
    Methods
    -------
    fit(X,y)
        Fits the models
    predict(X)
        Makes predictions
    """
    _EPS = 1e-09
    def __init__(self,n_alphas=100,l1_ratio=0.5,alpha_init=None,exp_y=1,exp_x=1):
        """
            n_alphas : int
                number of candidate values for the regularisation parameter in the final regressions
            l1_ratio : float
                elastic net mixing parameter,
                with 0<=l1_ratio<=1,
                where l1_ratio=0 leads to L2 (ridge)
                and l1_ratio=1 leads to L1 (lasso) penalisation
            alpha_init : ndarray, default=None
                vector of length q_targets
                containing the regularisation parameter
                for the initial regressions
                (if None: optimisation by cross-validation)
            exp_y : float, default=1
                non-negative number for exponentiating the target-target correlation coefficients
            exp_x : float, default=1
                non-negative number for exponentiating the feature-feature correlation coefficients
        """
        self.n_alphas = n_alphas
        self.l1_ratio = l1_ratio
        self.alpha_init = alpha_init
        self.exp_y = exp_y
        self.exp_x = exp_x
        self.n_ = self.p_ = self.q_ = None # dimensionality
        self.n_features_in_ = None # compatibility
        self.mu_y_ = self.sd_y_ = None # standardisation
        self.alpha_init_ = self.weight_ = self.model_ = None # modelling
    def fit(self,X:np.ndarray,y:np.ndarray,Z:np.ndarray|None=None) -> "CoopLasso": # pylint: disable=invalid-name
        """
        Fit CoopLasso
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, p_features) or (n_samples, p_features, q_targets)
            common feature matrix for all targets or a separate feature matrix for each target
        y : ndarray of shape (n_samples, q_targets)
            target matrix
        Z : ndarray of shape (p_features,) or (q_targets, p_features), or None
            logical vector or matrix
            indicating primary (1, True)
            and auxiliary features (0, False)
            for all targets together or each target separately
          
        Returns
        -------
        
        self: CoopLasso
            fitted model
        """
        if y.ndim==1:
            y = y.reshape(-1,1)
        check_array(array=X,allow_nd=True)
        check_array(array=y)
        self.n_, self.p_, self.q_ = _check_dims(X=X,y=y,Z=Z)
        self.n_features_in_ = self.p_
        if Z is None:
            Z = np.full((self.p_,self.q_),1)
        elif Z.ndim==1:
            Z = np.broadcast_to(Z[:,None],(self.p_,self.q_))
        self.mu_y_ = np.mean(y,axis=0)
        self.sd_y_ = np.std(y,axis=0)
        y = (y - self.mu_y_)/self.sd_y_
        cor_y = np.corrcoef(rankdata(y, axis=0),rowvar=False)
        # This would not return a matrix under q=2:
        # cor_y = spearmanr(y).statistic
        cor_y = np.nan_to_num(cor_y,nan=0)
        if X.ndim==2:
            cor = spearmanr(X).statistic
            cor = np.nan_to_num(cor,nan=0)
            cor_x = [cor] * self.q_
        elif X.ndim==3:
            cor_x = []
            for j in range(self.q_):
                cor = spearmanr(X[:,:,j]).statistic
                cor = np.nan_to_num(cor,nan=0)
                cor_x.append(cor)
        coef = np.full((self.p_, self.q_), np.nan)
        if self.l1_ratio is None:
            raise NotImplementedError(
                "Initial correlation coefficients (l1_ratio=None)"
                "have not yet been implemented."
            )
        else:
            if self.alpha_init is None:
                self.alpha_init_ = np.full(self.q_,np.nan)
                for j in range(self.q_):
                    enet = ElasticNetCV(l1_ratio=self.l1_ratio)
                    if X.ndim==2:
                        enet.fit(X,y[:,j])
                    else:
                        enet.fit(X[:,:,j],y[:,j])
                    coef[:,j] = enet.coef_
                    self.alpha_init_[j] = enet.alpha_
                # Alternative with multivariate initialisation:
                # enet = MultiTaskElasticNetCV(l1_ratio=l1_ratio)
                # enet.fit(X,y)
                # coef = enet.coef_.T
                # self.alpha_init_ = enet.alpha_
            else:
                self.alpha_init_ = self.alpha_init
                for j in range(self.q_):
                    enet = ElasticNet(alpha=self.alpha_init_[j],l1_ratio=self.l1_ratio)
                    if X.ndim==2:
                        enet.fit(X,y[:,j])
                    else:
                        enet.fit(X[:,:,j],y[:,j])
                    coef[:,j] = enet.coef_
                # Alternative with multivariate initialisation:
                # enet = MultiTaskElasticNet(alpha=alpha_init,l1_ratio=l1_ratio)
                # enet.fit(X,y)
                # coef = enet.coef_.T
        self.weight_ = []
        self.model_ = []
        xx = None
        if X.ndim==2:
            xx = np.hstack([X,-X])
        for i in range(self.q_):
            if X.ndim==3:
                xx = np.hstack([X[:,:,i],-X[:,:,i]])
            temp = (
                coef *
                (np.sign(cor_y[:, i]) *
                (np.abs(cor_y[:, i])**self.exp_y))
            )
            w_pos = np.full(self.p_,np.nan)
            w_neg = np.full(self.p_,np.nan)
            w_abs = np.full(self.p_,np.nan)
            for j in range(self.p_):
                cont = (
                    temp *
                    (np.sign(cor_x[i][:, j]) *
                    (np.abs(cor_x[i][:, j])**self.exp_x))[:, np.newaxis]
                )
                w_pos[j] = np.sum(np.maximum(cont, 0))
                w_neg[j] = np.sum(np.maximum(-cont, 0))
                w_abs[j] = np.sum(np.abs(cont))
            exclude = Z.T[i,:]==0
            w_pos[exclude] = 0
            w_neg[exclude] = 0
            weight = np.append(w_pos+self._EPS,w_neg+self._EPS)
            # This alternative does not need the non-negativity constraint:
            # weight = (w_abs + 1e-9)
            xx_scale = xx * weight
            model = lasso_path(X=xx_scale,y=y[:,i],n_alphas=self.n_alphas,alphas=None,positive=True)
            # This alternative is computationally expensive:
            # lasso_path(X=xx_scale,y=y[:,i],alphas=self.alphas[i]['model'][0],positive=True)
            self.weight_.append(weight)
            self.model_.append(model)
        return self
    def predict(self,X:np.ndarray,alpha:list[np.ndarray]|None=None) -> list[np.ndarray]: # pylint: disable=invalid-name
        """
        Make predictions
  
        Parameters
        ----------
        X : ndarray of shape (n_samples, p_features) or (n_samples, p_features, q_targets)
            common feature matrix for all targets,
            or a separate feature matrix for each target
        alpha : list of length q_targets or None, default=None
            one ndarray of non-negative regularisation parameters for each target;
            if None, predictions are returned for the fitted path
        
        Returns
        -------
        y_hat : list of length q_targets
            one ndarray of shape (n_samples, n_alphas) for each target
        
        """
        check_is_fitted(self,attributes=['model_'])
        y_hat = []
        #if X.ndim not in (2,3):
        #    raise ValueError("X must be a matrix or an array")
        newxx = None
        if X.ndim==2:
            newxx = np.hstack([X, -X])
        for i in range(len(self.model_)):
            if X.ndim==3:
                newxx = np.hstack([X[:,:,i],-X[:,:,i]])
            newx_scale = newxx * self.weight_[i]
            beta = self.model_[i][1]
            if alpha is None:
                y_hat.append((newx_scale @ beta)*self.sd_y_[i] + self.mu_y_[i])
            else:
                alpha_path = np.log(self.model_[i][0]+self._EPS)
                order_path = np.argsort(alpha_path)
                alpha_path = alpha_path[order_path]
                beta = beta[:,order_path]
                alpha_full = np.log(alpha[i]+self._EPS)
                # This avoids extrapolation outside range of fitted path:
                alpha_full = np.clip(alpha_full,alpha_path.min(),alpha_path.max())
                increase = np.hstack([True,np.diff(alpha_path)!=0])
                func_inter = interp1d(alpha_path[increase],beta[:,increase],axis=1)
                beta_inter = func_inter(alpha_full)
                y_hat.append((newx_scale @ beta_inter)*self.sd_y_[i] + self.mu_y_[i])
        return y_hat

class CoopLassoCV(BaseEstimator,RegressorMixin):
    """
    Cross-Validated Cooperative Multi-Task Lasso Regression
    
    Fits cooperative multi-task lasso regression,
    optimising the regularisation parameters by cross-validation.

    Parameters
    ----------
    cv : int, default=10
        number of cross-validation folds
    n_alphas : int, default=100
        number of candidate values for the regularisation parameter in the final regression
    l1_ratio : float, default=0.5
        elastic net mixing parameter for initial regression, ratio in [0,1]
    exp_y : float, default=1.0
        exponent for target-target correlation coefficients, non-negative number
    exp_x : float, default=1.0
        exponent for feature-feature correlation coefficients, non-negative number
    random_state : int or None, default=None
        random seed for determining the random fold identifiers
        
    Attributes
    ----------
    n_ : int
        number of training samples
    p_ : int
        number of features
    q_ : int
        number of targets
    model_ : list of length q_targets
        fitted models from CoopLasso
    alpha_ : list of length q_targets
        ndarray of regularisation parameters
    mse_ : list of length q_targets
        ndarray of cross-validated mean squared error for each value of alpha
    min_ : list of length q_targets
        indices of regularisation parameters corresponding to the lowest mean squared error
    coef_ : ndarray of shape (q_targets, p_features)
        estimated effects (of the feature in the column on the target in the row)
    
    Methods
    -------
    fit(X,y)
        Fits the models
    predict(X)
        Makes predictions
    """
    def __init__(self, cv=10, n_alphas=100, l1_ratio=0.5, exp_y=1, exp_x=1, random_state = None):
        """
            n_alphas : int
                number of candidate values for the regularisation parameter in the final regressions
            l1_ratio : float
                elastic net mixing parameter,
                with `0<=l1_ratio<=1`,
                where `l1_ratio=0` leads to L2 (ridge)
                and `l1_ratio=1` leads to L1 (lasso) penalisation
            alpha_init : ndarray, default=None
                vector of length q_targets
                containing the regularisation parameter
                for the initial regressions
                (if None: optimisation by cross-validation)
            exp_y : float, default=1
                non-negative number for exponentiating the target-target correlation coefficients
            exp_x : float, default=1
                non-negative number for exponentiating the feature-feature correlation coefficients
            random_state : int or None, default=None
                random seed for generating reproducible cross-validation folds
        """
        self.cv = cv
        self.n_alphas = n_alphas
        self.l1_ratio = l1_ratio
        self.exp_y = exp_y
        self.exp_x = exp_x
        self.random_state = random_state
        self.n_ = self.p_ = self.q_ = None # dimensionality
        self.n_features_in_ = None # compatibility
        self.alpha_ = self.mse_ = self.min_ = None # cross-validation
        self.model_ = self.coef_ = None # modelling
    def fit(self,X:np.ndarray,y:np.ndarray,Z:np.ndarray|None=None) -> "CoopLassoCV": # pylint: disable=invalid-name
        """
        Fit CoopLassoCV
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, p_features) or (n_samples, p_features, q_targets)
            common feature matrix for all targets or a separate feature matrix for each target
        y : ndarray of shape (n_samples, q_targets)
            target matrix
        Z : ndarray of shape (p_features,) or (q_targets, p_features), or None
            logical vector or matrix
            indicating primary (1, True) and auxiliary features (0, False)
            for all targets or each target
          
        Returns
        -------
        
        self : CoopLassoCV
            fitted model
        """
        if y.ndim==1:
            y = y.reshape(-1,1)
        check_array(array=X,allow_nd=True)
        check_array(array=y)
        self.n_, self.p_, self.q_ = _check_dims(X=X,y=y,Z=Z)
        self.n_features_in_ = self.p_
        self.model_ = CoopLasso(
            l1_ratio=self.l1_ratio,
            n_alphas=self.n_alphas,
            alpha_init=None,
            exp_y=self.exp_y,
            exp_x=self.exp_x)
        self.model_.fit(X=X,y=y,Z=Z)
        self.alpha_ = []
        for i in range(self.q_):
            self.alpha_.append(self.model_.model_[i][0])
        y_hat = np.full((self.n_,self.q_,self.n_alphas),np.nan)
        folds = KFold(n_splits=self.cv,shuffle=True,random_state=self.random_state)
        for train_id, test_id in folds.split(X=X,y=y):
            sub = CoopLasso(
                n_alphas=self.n_alphas,
                l1_ratio=self.l1_ratio,
                alpha_init=self.model_.alpha_init_,
                exp_y=self.exp_y,
                exp_x=self.exp_x)
            sub.fit(X=X[train_id,...], y=y[train_id,:], Z=Z)
            temp = sub.predict(X=X[test_id,...],alpha=self.alpha_)
            for j in range(len(temp)):
                y_hat[test_id,j,:] = temp[j]
        self.mse_ = []
        self.min_ = []
        self.coef_ = np.full((self.q_,self.p_),np.nan)
        for j in range(self.q_):
            mse = np.mean((y_hat[:,j,:] - y[:,j,np.newaxis])**2, axis=0)
            self.mse_.append(mse)
            id_min = np.argmin(mse)
            self.min_.append(id_min)
            temp = self.model_.model_[j][1][:,id_min]
            self.coef_[j,:] = temp[0:self.p_] - temp[self.p_:2*self.p_]
        return self
    def predict(self,X:np.ndarray) -> np.ndarray: # pylint: disable=invalid-name
        """
        Make predictions
  
        Parameters
        ----------
        X : ndarray of shape (n_samples, p_features) or (n_samples, p_features, q_targets)
            common feature matrix for all targets, or a separate feature matrix for each target
        
        Returns
        -------
        y_hat : ndarray of shape (n_samples, q_targets)
            matrix of predicted values (of the target in the column for the sample in the row)
        
        """
        check_is_fitted(self,attributes=['coef_'])
        y_hat = np.full((X.shape[0], self.q_), np.nan)
        newxx = None
        if X.ndim==2:
            newxx = np.hstack([X, -X])
        for i in range(self.q_):
            if X.ndim==3:
                newxx = np.hstack([X[:, :, i], -X[:, :, i]])
            newx_scale = newxx * self.model_.weight_[i]
            id_min = self.min_[i]
            beta = self.model_.model_[i][1][:, id_min]
            y_hat[:,i] = (newx_scale @ beta)*self.model_.sd_y_[i] + self.model_.mu_y_[i]
        return y_hat
