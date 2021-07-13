import jax
import jax.numpy as jnp
import numpy as np
from math import pi

from oed_local_linear import create_local_linear_funcs

def create_fun_and_map():
    def K(d):
        K = jnp.array([[d[0]**2, d[1]**(1/2)], \
                       [d[1] + 1, d[2]**(3/2)], \
                       [d[2]**(-1), d[3]**2]])
        return K
    def b(d):
        b = jnp.array([d[0]-1, d[1]*d[2], d[3]**2])
        return b
    # Linear forward model wrt theta:
    def fun(theta, d):
        theta = jnp.atleast_1d(theta.squeeze())
        return (jnp.einsum("ik,k->i", K(d), theta) + b(d)).squeeze()
    return (fun, K, b)

def create_likelihod_and_post(model, K_fun, b_fun, noise, inv_noise, mu_prior, inv_prior):
    k_theta = inv_prior.shape[0]
    k_y = inv_noise.shape[0]
    def posterior(theta, y, d):
        G1 = K_fun(d)
        b = b_fun(d)
        inv_cov = G1.T @ inv_noise @ G1 + inv_prior
        cov = jnp.linalg.inv(inv_cov)
        mu = ((y-b).T @ inv_noise @ G1 + mu_prior.T @ inv_prior) @ cov
        return - (1/2)*(k_theta)*jnp.log(2*pi) - (1/2)*jnp.log(jnp.linalg.det(cov)) \
            - (1/2)*(theta-mu).T @ inv_cov @ (theta-mu)
    def likelihood(theta, y, d):
        mean = model(theta, d)
        return - (1/2)*(k_y)*jnp.log(2*pi) - (1/2)*jnp.log(jnp.linalg.det(noise)) \
            - (1/2)*(y-mean).T @ inv_noise @ (y-mean)
    return (posterior, likelihood)

def create_lin_gradients(fun):
    model_funcs = {}
    fun_theta = jax.jacrev(fun, argnums=0)
    fun_d = jax.jacrev(fun, argnums=1)
    fun_2_theta = jax.jacrev(fun_theta, argnums=0)
    fun_d_theta = jax.jacrev(fun_theta, argnums=1)
    model_funcs["g"] = jax.vmap(fun, in_axes=(0,None))
    model_funcs["g_del_theta"] = jax.vmap(fun_theta, in_axes=(0,None))
    model_funcs["g_del_d"] = jax.vmap(fun_d, in_axes=(0,None), out_axes=0)
    model_funcs["g_del_2_theta"] = jax.vmap(fun_2_theta, in_axes=(0,None))
    model_funcs["g_del_d_theta"] = jax.vmap(fun_d_theta, in_axes=(0,None))
    return model_funcs

if __name__ == "__main__":
    # Specify noise and prior:
    noise_cov = jnp.diag(jnp.array([0.1, 0.1, 0.1]))
    prior_cov = jnp.diag(jnp.array([1.0, 1.0]))
    prior_mean = jnp.array([0., 0.])
    inv_noise = jnp.linalg.inv(noise_cov)
    inv_prior = jnp.linalg.inv(prior_cov)
    
    # Compute functions to calculate likelihood and posterior:
    fun, K, b = create_fun_and_map()

    # Create likelihood and posterior functions:
    posterior, likelihood = create_likelihod_and_post(fun, K, b, noise_cov, inv_noise, prior_mean, inv_prior)

    # Compute gradient of posterior and likelihoods wrt d: 
    post_fun = jax.vmap(posterior, in_axes=(0, 0, None))
    post_grad_fun = jax.vmap(jax.jacrev(posterior, argnums=2), in_axes=(0, 0, None))
    like_grad_fun = jax.vmap(jax.jacrev(likelihood, argnums=2), in_axes=(0, 0, None))

    # Compute gradients using local_linear.py implementations:
    theta_bounds = np.array([[-10., 10.], [-10., 10.]])
    model_funcs = create_lin_gradients(fun)
    log_probs_and_grads = \
    create_local_linear_funcs(model_funcs, noise_cov, prior_mean, prior_cov, theta_bounds, inv_noise=inv_noise, inv_prior=inv_prior)[2]
    
    # Compare results for random query points:
    num_samples = 10
    d = jnp.array(np.random.rand(4))
    query_y = jnp.array(np.random.rand(num_samples, 3))
    query_theta = jnp.array(np.random.rand(num_samples, 2))
    true_log_post, true_like_grad, true_post_grad = \
        post_fun(query_theta, query_y, d), like_grad_fun(query_theta, query_y, d), post_grad_fun(query_theta, query_y, d)
    log_post, log_like_grad, log_post_grad = log_probs_and_grads(d, query_theta, query_y)
    print(f"Log posterior max difference = {jnp.max(abs(true_log_post-log_post))}")
    print(f"Log likelihood gradient max difference = {jnp.max(abs(true_like_grad-log_like_grad))}")
    print(f"Log posterior gradient max difference = {jnp.max(abs(true_post_grad-log_post_grad))}")