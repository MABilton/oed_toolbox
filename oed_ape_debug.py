import jax
import jax.numpy as jnp
import numpy as np
from math import pi
from numpy.random import multivariate_normal as mvn

from oed_ape import find_optimal_d as find_ape_d
from oed_det import create_det_fun, find_optimal_d as find_det_d

def create_linear_model(K, b):
    def linear_model(theta, d):
        theta = jnp.atleast_1d(theta.squeeze())
        y = K(d) @ theta + b(d)
        return jnp.atleast_1d(y.squeeze())
    return linear_model

def get_lin_coeffs():
    def K(d):
        K = jnp.array([-1.*(d-5.)**2 + 20.])
        return K
    def b(d):
        b = jnp.array([0.])
        return b
    return (K, b)

def create_ape_funcs(model, model_vmap, K_fun, b_fun, noise_cov, prior_mean, prior_cov):
    inv_noise = jnp.linalg.inv(noise_cov) 
    inv_prior = jnp.linalg.inv(prior_cov)
    k_theta = prior_mean.size
    k_y = inv_noise.shape[0]
    def sample_prior(num_samples):
        prior_samples = mvn(prior_mean, prior_cov, size=num_samples)
        return prior_samples
    def sample_likelihood(d, prior_samples):
        n_samples = prior_samples.shape[0]
        means = model_vmap(prior_samples, d).reshape(n_samples, k_y)
        samples = mvn(np.zeros(k_y), noise_cov, size=n_samples)
        return samples + means
    def log_posterior(theta, y, d):
        G1 = K_fun(d)
        b = b_fun(d)
        inv_cov = G1.T @ inv_noise @ G1 + inv_prior
        cov = jnp.linalg.inv(inv_cov)
        mu = ((y-b).T @ inv_noise @ G1 + prior_mean.T @ inv_prior) @ cov
        return - (1/2)*(k_theta)*jnp.log(2*pi) - (1/2)*jnp.log(jnp.linalg.det(cov)) \
            - (1/2)*(theta-mu).T @ inv_cov @ (theta-mu)
    def log_likelihood(theta, y, d):
        mean = model(theta, d)
        return - (1/2)*(k_y)*jnp.log(2*pi) - (1/2)*jnp.log(jnp.linalg.det(noise_cov)) \
            - (1/2)*(y-mean).T @ inv_noise @ (y-mean)
    return (sample_prior, sample_likelihood, log_posterior, log_likelihood)

def create_log_probs_and_grads(log_post_fun, log_like_grad_fun, log_post_grad_fun):
    def log_probs_and_grads(d, prior_samples, like_samples):
        log_post = log_post_fun(prior_samples, like_samples, d)
        log_like_grad = log_like_grad_fun(prior_samples, like_samples, d)
        log_post_grad = log_post_grad_fun(prior_samples, like_samples, d)
        return (log_post, log_like_grad, log_post_grad)
    return log_probs_and_grads

if __name__ == "__main__":
    # Define linear model:
    K, b = get_lin_coeffs()
    lin_model = create_linear_model(K, b)

    # Define noise and prior:
    noise_cov = jnp.array([[0.1]])
    prior_cov = jnp.array([[1.]])
    prior_mean = jnp.array([0.])
    inv_noise = jnp.linalg.inv(noise_cov) 
    inv_prior = jnp.linalg.inv(prior_cov)
    
    # Define bounds for theta and d:
    theta_bounds = jnp.array([[-10., 10.]])
    d_bounds = jnp.array([[0., 10.]])

    # Compute required ape functions and gradients + vectorise if required:
    model_vmap = jax.vmap(lin_model, in_axes=(0,None))
    sample_prior, sample_likelihood, log_posterior, log_likelihood \
         = create_ape_funcs(lin_model, model_vmap, K, b, noise_cov, prior_mean, prior_cov)
    log_post_vmap = jax.vmap(log_posterior, in_axes=(0,0,None))
    log_like_grad = jax.vmap(jax.jacrev(log_likelihood, argnums=2), in_axes=(0,0,None))
    log_post_grad = jax.vmap(jax.jacrev(log_posterior, argnums=2), in_axes=(0,0,None))
    # Place log probabilities and grads in single function:
    log_probs_and_grads = create_log_probs_and_grads(log_post_vmap, log_like_grad, log_post_grad)

    # Call APE minimisation script:
    ape_d = find_ape_d(sample_likelihood, sample_prior, log_probs_and_grads, d_bounds)

    # Find D-optimal:
    model_funcs = {}
    model_funcs["g_del_theta"] = K
    model_funcs["g_del_d_theta"] = jax.jacrev(K, argnums=0)
    det_and_grad = create_det_fun(model_funcs, inv_noise, inv_prior)
    det_d = find_det_d(det_and_grad, d_bounds)
    print(f"D-optimal design = {det_d}, APE Optimal Design = {ape_d}")