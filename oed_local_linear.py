from math import pi
import numpy as np
from numpy import linalg
from oed_map import create_compute_map

# Functions to be specified by user stored in "model_funcs" dictionary:
#   - g(theta, d) = Forward model which returns (N_samples, N_theta) array
#   - g_del_theta(theta, d) = grad of g wrt theta, returning (N_samples, N_y, N_theta) array
#   - g_del_d(theta, d) = grad of g wrt d, returning (N_samples, N_y, N_d) array
#   - g_del_2_theta(theta, d) = 2nd order grad of g wrt d, returning (N_samples, N_y, N_theta, N_theta) array
#   - g_del_d_theta(theta, d) = mixed grad of g wrt d and theta, returning (N_samples, N_y, N_theta, N_d) array
# Note: theta.shape = (N_samples, N_theta), d.shape = (N_d,)

# Compute eps for numerical differentiation if required:
eps = 7*np.sqrt(np.finfo(np.float64).eps)

# Add error flag for d_dim=None if need to numerically differentiate:
def create_local_linear_funcs(model_funcs, noise_cov, prior_mean, prior_cov, theta_bounds, d_dim, approx_flag=True, dx=eps, vect_flag=False):
    # Compute inverse of noise and prior matrices if required:
    inv_noise, inv_prior = linalg.inv(noise_cov), linalg.inv(prior_cov)
    # Note dimensions of problem:
    y_dim, theta_dim = noise_cov.shape[0], prior_mean.size
    # Create 'vectorised' version of model functions if required:
    if vect_flag:
        model_funcs = create_vectorised_model(model_funcs, y_dim)
    # Create numerical gradient functions if required:
    for grad in ["g_del_theta", "g_del_d", "g_del_2_theta", "g_del_d_theta"]:
        if grad not in model_funcs:
            model_funcs[grad] = create_numerical_grad(model_funcs["g"], grad, y_dim, theta_dim, d_dim, dx)
    # Create function to compute log_post and gradients of log_post and log_like:
    log_probs_and_grads = create_log_probs_and_grads(model_funcs, inv_noise, prior_mean, prior_cov, inv_prior, theta_bounds, d_dim, approx_flag)
    return log_probs_and_grads

#
#   CREATE LOG PROB FUNCTIONS
#

def create_log_post_and_grad(inv_noise, prior_mean, prior_cov, inv_prior):
    # Computes mean, covariance matrix and inverse covariance matrix of posterior:
    def compute_mean_and_cov(y, b, G1):
        # Compute inverse cov and cov matrices:
        inv_cov = np.einsum("aki,kl,alj->aij", G1, inv_noise, G1) + inv_prior
        cov = linalg.inv(inv_cov)
        # Compute mean:
        mean = np.einsum("aj,jk,aki->ai", (y-b), inv_noise, G1) + prior_mean.T @ inv_prior
        mean = np.einsum("ak,aki->ai", mean, cov)
        return (mean, cov, inv_cov)

    # Computes gradient of posterior mean, poterior covariance matrix, and inverse of poterior covariance matrix:
    def compute_mean_and_cov_grad(y, b, theta_map, map_grad, cov, G1, G2, G12):
        # Grad of inverse covariance matrix - same value?:
        inv_cov_grad = np.einsum("alik,lm,amj->aijk", G12, inv_noise, G1) + np.einsum("ali,lm,amjk->aijk", G1, inv_noise, G12)
        # Grad of covariance matrix:
        cov_grad = -1*np.einsum("ail,almk,amj->aijk", cov, inv_cov_grad, cov)
        # Grad of posterior mean:
        b_grad = np.einsum("akj,aik->aij", map_grad, G1) + G2 \
             - np.einsum("aikj,ak->aij", G12, theta_map) - np.einsum("aik,akj->aij", G1, map_grad)
        mean_grad = np.einsum("akij,al,lm,amk->aij", cov_grad, y-b, inv_noise, G1) \
            - np.einsum("aki,alj,lm,amk->aij", cov, b_grad, inv_noise, G1) \
                + np.einsum("aki,al,lm,amkj->aij", cov, y-b, inv_noise, G12) \
                 + np.einsum("l,lk,akij->aij", prior_mean, prior_cov, cov_grad)
        return (mean_grad, cov_grad, inv_cov_grad)

    # Computes log probability of posterior and gradient of log posterior wrt d:
    def log_post_and_grad(theta, y, theta_map, map_grad, G, G1, G2, G12):
        # Compute b linearisation coefficient:
        b = G - np.einsum("aij,aj->ai", G1, theta_map)
        # Compute mean and covariance matrix of posterior:
        mean, cov, inv_cov = compute_mean_and_cov(y, b, G1)
        # Compute theta - posterior mean:
        del_theta = theta - mean
        # Compute log posterior probability:
        log_post = -0.5*(theta.shape[1])*np.log(2*pi) - 0.5*np.log(linalg.det(cov)) \
            - 0.5*np.einsum("ai,aij,aj->a", del_theta, inv_cov, del_theta)
        # Compute gradient of posterior mean and cov wrt d:
        mean_grad, cov_grad, inv_cov_grad = \
             compute_mean_and_cov_grad(y, b, theta_map, map_grad, cov, G1, G2, G12)
        # Compute gradient of log posterior:
        det_grad = np.einsum("aijk,aji->ak", cov_grad, inv_cov)
        quad_grad = np.einsum("aijk,ai,aj->ak", inv_cov_grad, del_theta, del_theta) \
               - 2*np.einsum("alk,ali,ai->ak", mean_grad, inv_cov, del_theta)
        log_post_grad = -0.5*det_grad - 0.5*quad_grad
        return (log_post, log_post_grad)
    return log_post_and_grad

def create_log_probs_and_grads(model_funcs, inv_noise, prior_mean, prior_cov, inv_prior, theta_bounds, d_dim, approx_flag):
    # Unpack functions:
    g = model_funcs["g"]
    g_del_theta = model_funcs["g_del_theta"]
    g_del_d = model_funcs["g_del_d"]
    g_del_d_theta = model_funcs["g_del_d_theta"]
    
    # Create helper functions:
    compute_map = create_compute_map(model_funcs, inv_noise, prior_mean, inv_prior, theta_bounds, d_dim)
    log_post_and_grad = create_log_post_and_grad(inv_noise, prior_mean, prior_cov, inv_prior)

    def log_probs_and_grads(d, theta_samples, y_samples):
        
        # Initialise MAP estimates to prior sample?
        theta_init = theta_samples if approx_flag else None

        # Compute MAP:
        map_vals, map_grads = compute_map(d, y_samples, theta_samples, theta_init)
        
        # Compute required model-based arrays:
        G = g(map_vals, d)
        G1 = g_del_theta(map_vals, d)
        G2 = g_del_d(map_vals, d)
        G12 = g_del_d_theta(map_vals, d)
        
        # Compute gradient of log likelihood:
        G_like = g(theta_samples, d)
        G2_like = g_del_d(theta_samples, d)
        log_like_grad = np.einsum("ajk,jl,al->ak", G2_like, inv_noise, y_samples-G_like)

        # Compute log posterior probability and gradient of log posterior:
        log_post, log_post_grad = log_post_and_grad(theta_samples, y_samples, map_vals, map_grads, G, G1, G2, G12)

        return (log_post, log_like_grad, log_post_grad)
    return log_probs_and_grads

#
#   CREATE NUMERICAL GRADIENTS
#

def create_numerical_grad(model, grad, y_dim, theta_dim, d_dim, dx):    
    model_grad, fun_dim = model, (y_dim,)
    if "theta" in grad:
        model_grad = nd_wrt_theta(model_grad, dx, theta_dim, fun_dim)
        fun_dim += (theta_dim,)
        if "del_2_theta" in grad:
            model_grad = nd_wrt_theta(model_grad, dx, theta_dim, fun_dim)
    if "del_d" in grad:
        model_grad = nd_wrt_d(model_grad, dx, d_dim, fun_dim)
    return model_grad

def nd_wrt_theta(func, eps_theta, theta_dim, fun_dim):
    def diff_theta(theta, d):
        # Compute gradients:
        theta_in, d_in = np.repeat(theta, theta_dim, axis=0), np.repeat(d, theta_dim, axis=0)
        num_samples = theta.shape[0]
        del_theta = eps_theta*np.tile(np.eye(theta_dim), (num_samples,1))
        y_1 = func(theta_in-del_theta, d_in)
        y_2 = func(theta_in+del_theta, d_in)
        grad = (y_2 - y_1)/(2*eps_theta)
        # Reshape gradient to correct shape:
        perm = np.fromiter((i + j*theta_dim for i in range(0,theta_dim) for j in range(0,num_samples)), dtype=np.int32)
        grad = grad[perm,:].reshape((theta_dim,)+(num_samples,)+fun_dim, order="C")
        grad = np.rollaxis(grad, 0, grad.ndim)
        return grad
    return diff_theta

def nd_wrt_d(func, eps_d, d_dim, fun_dim):
    def diff_d(theta, d):
        # Compute gradients:
        theta_in, d_in = np.repeat(theta, d_dim, axis=0), np.repeat(d, d_dim, axis=0)
        num_samples = theta.shape[0]
        # del_d = eps_d*np.repeat(np.eye(d_dim), num_samples, axis=0)
        del_d = eps_d*np.tile(np.eye(d_dim), (num_samples,1))
        y_1 = func(theta_in, d_in-del_d)
        y_2 = func(theta_in, d_in+del_d)
        grad = (y_2 - y_1)/(2*eps_d)
        # Reshape gradient to correct shape:
        perm = np.fromiter((i + j*d_dim for i in range(0,d_dim) for j in range(0,num_samples)), dtype=np.int32)
        grad = grad[perm,:].reshape((d_dim,)+(num_samples,)+fun_dim, order="C")
        grad = np.rollaxis(grad, 0, grad.ndim)
        return grad
    return diff_d

#
#   VECTORISE MODEL FUNCTION
#

def create_vectorised_model(model_funcs, dim_y):
    for grad in ["g_del_theta", "g_del_d", "g_del_2_theta", "g_del_d_theta"]:
        if grad not in model_funcs:
            model_funcs[grad] = create_vectorised_fun(model_funcs[grad], dim_y)
    return model_funcs

def create_vectorised_fun(fun, dim_y):
    def vectorised_fun(theta, d):
        y = []
        num_samples = theta.shape[0]
        for i in range(num_samples):
            y.append(fun(theta[i,:], d[i,:]))
        return np.array(y).reshape((num_samples, dim_y))
    return vectorised_fun