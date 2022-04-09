from math import pi
import numpy as np
from scipy.stats import multivariate_normal as mvn

#
#   Function Calls and Pre-Processing
#

def _attempt_func_call(func, outputs_dict, args, func_key):
    if func is not None:
        outputs_dict[func_key] = func(*args)
    else:
        raise AttributeError(f'{func_key} function has not been specified.')
    return outputs_dict 

def _preprocess_inputs(**inputs):
    inputs = _ensure_arrays_are_2d(inputs)
    inputs = _check_batch_dimension(inputs)
    outputs = tuple(inputs.values())
    return outputs if len(outputs) > 1 else outputs[0]

def _ensure_arrays_are_2d(inputs):
    for key, val in inputs.items():
        val = np.atleast_1d(val)
        if val.ndim == 1:
            inputs[key] = val[None,:]
        elif val.ndim > 2:
            raise ValueError(f'{key} must be a one or two-dimensional array.')
    return inputs

def _check_batch_dimension(inputs):
    num_batch = np.max([val.shape[0] for val in inputs.values()])
    for key, val in inputs.items():
        if val.shape[0] == 1:
            inputs[key] = np.broadcast_to(val, shape=(num_batch, val.shape[-1]))
        elif val.shape[0] != num_batch:
            raise ValueError(f'Expected {key} input to have a batch dimension of {num_batch}; ' 
                                f'instead, it was {val.shape[0]}.')
    return inputs

#
#   Control Variates
#

def apply_control_variates(val, *cv_list, **cv_dict):
    # Add singleton dimension if val=loss
    for _ in range(2-val.ndim):
        val = val[:, None] 
    cv_vec = []
    for cv in [*cv_list, *cv_dict.values()]:
        num_samples, size = cv.shape[0], np.prod(cv.shape[1:], dtype=int)
        cv_vec.append(cv.reshape(num_samples, size))
    cv_vec = np.concatenate(cv_vec, axis=1)
    val_vec = val.reshape(val.shape[0], np.prod(val.shape[1:], dtype=int))
    cv_var = _compute_covariance(cv_vec, cv_vec)
    val_cv_cov = _compute_covariance(cv_vec, val_vec-np.mean(val_vec,axis=0))
    a = _solve_for_a(cv_var, val_cv_cov)
    val_vec = np.mean(val_vec - np.einsum("ji,aj->ai", a, cv_vec), axis=0)
    return val_vec.reshape(val.shape[1:])

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

#
#   Gaussian Functions
#

def gaussian_sample(num_samples, mean, cov=None, cov_chol=None, rng=None, is_batched=False):
    mean, cov, cov_chol = _reshape_mean_and_cov(is_batched, mean, cov, cov_chol)
    cov_chol = _get_cov_chol(cov, cov_chol)
    epsilon = unit_gaussian_sample(mean.shape[-1], num_samples, rng) # (num_samples, mean_dim)
    return gaussian_transform(epsilon, mean, cov_chol=cov_chol) # (num_batch, num_samples, mean_dim) OR (num_samples, mean_dim)

def gaussian_transform(epsilon, mean, cov=None, cov_chol=None):
    is_batched = True
    mean, cov, cov_chol = _reshape_mean_and_cov(is_batched, mean, cov, cov_chol)
    cov_chol = _get_cov_chol(cov, cov_chol)
    return mean + np.einsum('aij,aj->ai', cov_chol, epsilon)

def _get_cov_chol(cov, cov_chol):
    if (cov is None) and (cov_chol is None):
        raise ValueError('Must specify either cov or cov_chol.')
    if cov_chol is None:
        cov_chol = np.linalg.cholesky(cov)
    return cov_chol

def unit_gaussian_sample(ndim, num_samples, rng):
    return mvn.rvs(mean=np.zeros(ndim), cov=np.identity(ndim), size=num_samples, random_state=rng).reshape(num_samples, ndim)

def gaussian_logpdf(x, mean, cov, icov=None):
    if icov is None:
        icov = np.lingalg.inv(cov)
    is_batched = True
    mean, cov, icov = _reshape_mean_and_cov(is_batched, mean, cov, icov)
    x_dim = mean.shape[-1]
    return -0.5*x_dim*np.log(2*pi) - 0.5*np.einsum('ai,aij,aj->a', x-mean, icov, x-mean) - 0.5*np.log(np.linalg.det(cov))

def _reshape_mean_and_cov(is_batched, mean, *cov_tuple):
    cov_list = list(cov_tuple)
    mean_ndim = 2 if is_batched else 1
    cov_ndim = 3 if is_batched else 2
    for _ in range(mean_ndim-mean.ndim):
        mean = mean[None,:]
    for idx, cov in enumerate(cov_list):
        if cov is not None:
            for _ in range(cov_ndim-cov.ndim):
                cov_list[idx] = cov[None,:]
    return (mean, *cov_list)