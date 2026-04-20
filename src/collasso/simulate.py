"""
Sparse linear multi-task regression 
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
from scipy.stats import multivariate_normal
from docrep import DocstringProcessor

docstrings = DocstringProcessor()
#docstrings['parameters'] = """x"""

#--- simulate data ---

@docstrings.get_sections(base='simulate',sections=['Parameters']) # pylint: disable=no-value-for-parameter
@docstrings.dedent
def simulate(
    *,
    n0:int=100,
    n1:int=10000,
    p:int=200,
    q:int=10,
    rho:float=0.90,
    kappa:float=1.00,
    prob_com:float=0.05,
    prob_sep:float=0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # pylint: disable=too-many-arguments,too-many-locals
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
        correlation coefficient, `0<=rho<=1`
    kappa : float, default=1.00
        correlation coefficient, `0<=kappa<=1`
    prob_com : float, default=0.05
        probability of common effects for all targets, 0<=prob_com<=1
    prob_sep : float, default=0.05
        probability of separate effects for each target
        
    Raises
    ------
    ValueError
      
    Returns
    -------
    x_train : ndarray of shape (n0_samples,p_features) or (n0_samples,p_features,q_targets)
        training feature matrix or matrices,
        common matrix for all targets (if kappa=1)
        or separate matrix for each target (if 0<=kappa<1)
    y_train : ndarray of shape (n0_samples,q_targets)
        training target matrix
    x_test : ndarray of shape (n1_samples,p_features) or (n1_samples,p_features,q_targets)
        test feature matrix or matrices,
        common matrix for all targets (if kappa=1)
        or separate matrix for each target (if 0<=kappa<1)
    y_test : ndarray of shape (n1_samples,q_targets)
        test target matrix
    beta : ndarray of shape (p_features,q_targets)
        true effects in the training and the test data
        (of the feature in the row on the target in the column)
    """
    if n0 < 10:
        raise ValueError(f"Use n0>=10 (not n0={n0})")
    if n1 < 10:
        raise ValueError(f"Use n1>=10 (not n1={n1})")
    if p < 2:
        raise ValueError(f"Use p>=2 (not p={p})")
    if q < 2:
        raise ValueError(f"Use q>=2 (not q={q})")
    if not 0 <= rho <= 1:
        raise ValueError(f"Use rho in [0, 1] (not rho={rho})")
    if not 0 <= kappa <= 1:
        raise ValueError(f"Use kappa in [0, 1] (not kappa={kappa})")
    if not 0 <= prob_com <= 1:
        raise ValueError(f"Use prob_com in [0, 1] (not prob_com={prob_com})")
    if not 0 <= prob_sep <= 1:
        raise ValueError(f"Use prob_sep in [0, 1] (not prob_sep={prob_sep})")
    n = n0 + n1
    fold = np.array([0]*n0+[1]*n1)
    x = _simulate_features(n=n,p=p,q=q,rho=rho,kappa=kappa)
    beta = _simulate_effects(p=p,q=q,prob_com=prob_com,prob_sep=prob_sep)
    y = _simulate_targets(n=n,q=q,x=x,beta=beta)
    x_train, y_train = x[fold==0,...], y[fold==0]
    x_test, y_test = x[fold==1,...], y[fold==1]
    return x_train, y_train, x_test, y_test, beta

docstrings.keep_params('simulate.parameters', 'p', 'q', 'rho', 'kappa')

@docstrings.dedent
def _simulate_features(*,n:int,p:int,q:int,rho:float,kappa:float) -> np.ndarray:
    """
    Simulate Features
    
    Parameters
    ----------
    n : int
        number of samples
    %(simulate.parameters.p|q|rho|kappa)s
      
    Returns
    -------
    x : ndarray of shape (n_samples, p_features) if kappa=1
        or (n_samples, p_features, q_targets) if 0<=kappa<1
    """
    mean = np.zeros(p)
    idx = np.arange(p)
    row_idx, col_idx = np.meshgrid(idx, idx)
    sigma = rho ** np.abs(col_idx - row_idx)
    x_base = multivariate_normal.rvs(mean=mean,cov=sigma,size=n)
    if kappa==1:
        x = x_base
    else:
        x = np.full((n,p,q),np.nan)
        for k in range(q):
            noise = multivariate_normal.rvs(mean=mean,cov=sigma,size=n)
            x[:,:,k] = np.sqrt(kappa)*x_base + np.sqrt(1-kappa)*noise
    return x

docstrings.keep_params('simulate.parameters', 'p', 'q', 'prob_com', 'prob_sep')

@docstrings.dedent
def _simulate_effects(*,p:int,q:int,prob_com:float,prob_sep:float) -> np.ndarray:
    """
    Simulate Effects
    
    Parameters
    ----------
    %(simulate.parameters.p|q|prob_com|prob_sep)s
    
    Returns
    -------
    beta : ndarray of shape (p_features, q_targets)
    """
    beta_com = (
        np.random.binomial(n=1, p=prob_com, size=p) *
        np.abs(np.random.normal(size=p))
    )
    beta_sep = (
        np.random.binomial(n=1, p=prob_sep, size=p * q).reshape(p, q) *
        np.abs(np.random.normal(size=p * q)).reshape(p, q)
    )
    beta = beta_com[:, np.newaxis] + beta_sep
    return beta

docstrings.keep_params('simulate.parameters', 'n', 'q')

@docstrings.dedent
def _simulate_targets(*,n:int,q:int,x:np.ndarray,beta:np.ndarray):
    """
    Simulate Targets
    
    Parameters
    ----------
    x : np.ndarray of shape (n_samples,p_features) or (n_samples,p_features,q_targets)
        common feature matrix or separate feature matrices
    beta: np.ndarray of shape (p_features,q_targets)
        effect matrix
    %(simulate.parameters.n|q)s
    
    Returns
    -------
    y : ndarray of shape (n_samples,q_targets)
    """
    if x.ndim==2:
        eta = x @ beta
    else:
        eta = np.full((n,q),np.nan)
        for k in range(q):
            eta[:,k] = x[:,:,k] @ beta[:,k]
    noise_sd = 0.5 * np.std(eta, axis = 0)
    y = eta + np.random.normal(size = eta.shape, scale = noise_sd)
    return y
