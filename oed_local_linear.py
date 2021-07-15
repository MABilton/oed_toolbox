from math import inf, pi
import numpy as np
from numpy.random import multivariate_normal as mvn
from numpy import linalg
from oed_map import create_compute_map

# Set seed for reproducability:
np.random.seed(42)

# Functions to be specified by user stored in "model_funcs" dictionary:
#   - g(theta, d) = Forward model which returns (N_samples, N_theta) array
#   - g_del_theta(theta, d) = grad of g wrt theta, returning (N_samples, N_y, N_theta) array
#   - g_del_d(theta, d) = grad of g wrt d, returning (N_samples, N_y, N_d) array
#   - g_del_2_theta(theta, d) = 2nd order grad of g wrt d, returning (N_samples, N_y, N_theta, N_theta) array
#   - g_del_d_theta(theta, d) = mixed grad of g wrt d and theta, returning (N_samples, N_y, N_theta, N_d) array
# Note: theta.shape = (N_samples, N_theta), d.shape = (N_d,)

def create_local_linear_funcs(model_funcs, noise_cov, prior_mean, prior_cov, theta_bounds, inv_noise=None, inv_prior=None):
    # Create prior function to sample from:
    sample_prior = create_sample_prior(prior_mean, prior_cov)
    # Create likelihood function to sample from:
    sample_likelihood = create_sample_likelihood(model_funcs["g"], noise_cov)
    # Compute inverse of noise and prior matrices if required:
    if inv_noise is None:
        inv_noise = linalg.inv(noise_cov)
    if inv_prior is None:
        inv_prior = linalg.inv(prior_cov)
    # Create function to compute log_post and gradients of log_post and log_like:
    log_probs_and_grads = create_log_probs_and_grads(model_funcs, inv_noise, prior_mean, prior_cov, inv_prior, theta_bounds)
    return (sample_prior, sample_likelihood, log_probs_and_grads)

#
#   CREATE SAMPLING FUNCTIONS
#

def create_sample_prior(prior_mean, prior_cov):
    def sample_prior(num_samples):
        return mvn(prior_mean, prior_cov, size=num_samples)
    return sample_prior

# See https://juanitorduz.github.io/multivariate_normal/ for theory:
def create_sample_likelihood(model, noise_cov):
    # Sample from zero mean, unit variance Gaussian, then transform for each sample:
    def sample_likelihood(d, theta_samples):
        # Compute means -> means.shape = (n_samples, y_dim):
        means = model(theta_samples, d)
        # Compute number of zero mean Gaussian samples to draw -> samples.shape = (n_samples, y_dim):
        y_dim, n_samples = means.shape[1], means.shape[0]
        samples = mvn(np.zeros(y_dim), noise_cov, size=n_samples)
        # Transform samples:
        return samples + means
    return sample_likelihood

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

def create_log_probs_and_grads(model_funcs, inv_noise, prior_mean, prior_cov, inv_prior, theta_bounds):
    # Unpack functions:
    g = model_funcs["g"]
    g_del_theta = model_funcs["g_del_theta"]
    g_del_d = model_funcs["g_del_d"]
    g_del_d_theta = model_funcs["g_del_d_theta"]
    
    # Create helper functions:
    compute_map = create_compute_map(model_funcs, inv_noise, prior_mean, inv_prior, theta_bounds)
    log_post_and_grad = create_log_post_and_grad(inv_noise, prior_mean, prior_cov, inv_prior)

    def log_probs_and_grads(d, theta_samples, y_samples):
        # Compute MAP:
        map_vals, map_grads = compute_map(d, y_samples)

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