from math import pi
import numpy as np
from scipy.stats import multivariate_normal as mvn

def apply_control_variates(val, *cv_list, **cv_dict):
    cv_vec = []
    for cv in [*cv_list, *cv_dict.values()]:
        num_samples, size = cv.shape[0], np.prod(cv.shape[1:], dtype=int)
        cv_vec.append(cv.reshape(num_samples, size))
    cv_vec = np.concatenate(cv_vec, axis=1)
    val_vec = val.reshape(val.shape[0], np.prod(val.shape[1:], dtype=int))
    cv_var = _compute_covariance(cv_vec, cv_vec)
    val_cv_cov = _compute_covariance(cv_vec, val_vec-np.mean(val_vec,axis=1))
    a = _solve_for_a(cv_var, val_cv_cov)
    val_vec = np.mean(val_vec - np.einsum("ij,aj->ai", a, cv_vec), axis=0)
    return val_vec.reshape(np.prod(val.shape[1:], dtype=int))

def _compute_covariance(vec1, vec2):
    return np.mean(np.einsum('ai,aj->aij', vec1, vec2), axis=0)

def _solve_for_a(cv_var, val_cv_cov, min_eps=1e-9, max_eps=1e-6, eps_increment=1e1):
    eps = 0
    solved = False
    while not solved:
        try:
            a = np.linalg.solve(cv_var + eps*np.identity(cv_var.shape[0]), val_cv_cov)
            solved = True
        except np.linalg.LinAlgError as error:
            if eps == 0:
                eps = min_eps
            elif eps < max_eps:
                eps *= eps_increment
            else:
                raise error
    return a

def gaussian_sample(num_samples, mean, cov, rng):
    mean, cov = _reshape_mean_and_cov(mean, cov)
    x_dim = mean.shape[-1]
    samples = mvn.rvs(mean=np.zeros(x_dim), cov=np.identity(x_dim), size=num_samples, random_state=rng).reshape(num_samples, x_dim)
    cholesky_cov = np.linalg.cholesky(cov)
    return mean + np.einsum('bji,aj->abi', cholesky_cov, samples) # shape = (num_samples, num_batch, dim_theta)

def gaussian_logpdf(x, mean, cov, icov=None):
    if icov is None:
        icov = np.lingalg.inv(cov)
    mean, cov, icov = _reshape_mean_and_cov(mean, cov, icov)
    x_dim = mean.shape[-1]
    return -0.5*x_dim*np.log(2*pi) - 0.5*np.einsum('ai,aij,aj->a', x-mean, icov, x-mean) - 0.5*np.log(np.linalg.det(cov))

def _reshape_mean_and_cov(mean, cov, icov=None):
    for _ in range(2-mean.ndim):
        mean = mean[None,:]
    for _ in range(3-cov.ndim):
        cov = cov[None,:]
    if icov is not None:
        for _ in range(3-icov.ndim):
            icov = icov[None,:]
    return (mean, cov) if icov is None else (mean, cov, icov)