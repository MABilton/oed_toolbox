import jax
import jax.numpy as jnp
import jax.scipy.linalg.det as jlinalg
import numpy as np
from numpy.random import multivariate_normal as mvn
from scipy.optimize import minimize
from math import inf, pi

# This will create FUNCTIONS which can then be passed to oed_optimise

def find_optimal_d(likelihood_fun, likelihood_grad_fun, noise_cov, prior_mean, prior_cov, d_bounds, num_samples=100, num_repeats=9):
    best_ape = inf
    # Compute required functions TODO:
    funcs = create_functions(likelihood_fun, likelihood_grad_fun)
    # Sample from prior:
    theta_samples = mvn(prior_mean, prior_cov, num_samples)
    for i in range(num_repeats):
        d_0 = np.random.uniform(d_bounds[0][0], d_bounds[0][1], size=1)
        d, ape = minimise_ape(d_0, theta_samples, noise_cov)
        if ape < best_ape:
            best_ape, best_d = ape, d
    return best_d

def minimise_ape(d_0, theta_samples, noise_cov, funcs):
    best_ape = inf
    # Initialise d:
    d = d_0
    for i in range(max_steps):
        # Draw samples from likelihood for current d:
        y_samples = sample_likelihood(d, theta_samples, noise_cov, model_fun)

        # Compute posterior MAP for all samples drawn:
        theta_map, map_jac = find_map(d, theta_samples, y_samples, map_loss, map_loss_grad)

        # Compute APE and grad of APE:
        ln_post, grad = compute_logpost_and_grad_vmap(theta, y, d, theta_map, map_jac)
        ape, ape_grad = jnp.mean(ln_post, axis=0), jnp.mean(grad, axis=0)

        # Perform stochastic step forward:
        d = update_d(d, ape_grad)
        
        # Store if best d found thus far:
        if ape < best_ape:
            best_ape, best_d = ape, d
    return (best_d, best_ape)

def sample_likelihood(d, theta_samples, noise_cov, model_fun):
    like_means = model_fun(theta_samples, d)
    like_samples = []
    for i in range(theta_samples.size[0]):
        like_samples.append(mvn(like_means[i,:], noise_cov, 1))
    like_samples = jnp.vstack(like_samples)
    return like_samples

def find_map(d, y_samples, theta_samples, map_loss_fun, map_loss_grad_fun, num_repeats=9):
    sample_map, sample_jac = [], []
    # Compute map for each y sample:
    for i in range(y_samples.shape[0]):
        y_i, theta_i = y_samples[i,:], theta_samples[i,:]
        map_i, jac_i = find_map_one_sample(d, y_i, theta_i, map_loss_fun, map_loss_grad_fun, num_repeats)
        sample_map.append(map_i)
        sample_jac.append(jac_i)
    sample_map, sample_jac = jnp.vstack(sample_map), jnp.vstack(sample_jac)
    return (sample_map, sample_jac)

def find_map_one_sample(d, y, theta_guess, map_loss, map_loss_grad, num_repeats, theta_bounds=None):
    # Assume call syntax of (theta, y, d) for map_loss and map_loss_grad
    best_loss = inf
    # Create required functions:
    for i in range(num_repeats):
        rand_num = mvn(jnp.ones(theta_guess.size), jnp.identity(theta_guess.size))
        theta_0 = rand_num*theta_guess
        opt_result = minimize(map_loss, theta_0, jac=map_loss_grad, args=(y,d), method="L-BFGS-B", bounds=theta_bounds)
        if opt_result['fun'] < best_loss:
            best_loss = opt_result['fun']
            best_map = opt_result['x']
            best_jac = opt_result['jac']
    return (best_map, best_jac)

def compute_logpost_and_grad(theta, y, d, theta_map, map_jac, model_grad_fun, map_loss_fun_grads):

    # Compute gradient of MAP point:
    map_grad = map_grad(theta_map, y, d, map_jac, map_loss_fun_grads)

    # Compute linearisation:
    G = compute_G(theta_map, d, model_grad_fun)
    b = compute_b(theta_map, d, model_grad_fun)

    # Compute gradient of linearisation wrt d
    G_grad = compute_G_grad(theta_map, d, map_grad, model_grad_fun, G_grad_args_fun)
    b_grad = compute_b_grad(theta_map, d, map_grad, model_grad_fun, b_grad_args_fun)

    # Compute derivatives of ln post and ln like:
    ln_post_grad = compute_log_post_grad(y, theta, G, b, inv_cov_y, inv_cov_theta, mean_theta)
    ln_like_grad = compute_log_like_grad(y, theta, G, b, cov_y, inv_cov_y)

    # Compute ln_post and ln_like prob values:
    ln_post = compute_log_post(y, theta, G, b, inv_cov_y, inv_cov_theta, mean_theta)

    # Compute gradient of APE:
    grad = ln_post*ln_like_grad + ln_post_grad
    return (ln_post, grad)

def map_grad(theta_map, y, d, map_jac, map_loss_fun_grads):
    return jlinalg.solve(map_loss_fun_grads[1](theta_map, y, d), map_jac-map_loss_fun_grads[0](theta_map, y, d))

def compute_log_like(y, theta, G, b, cov_y, inv_cov_y):
    k = cov_y.shape[0]
    mean = G @ theta + b
    return -k/2*jnp.log(2*pi) + -1/2*jnp.log(jlinalg.det(cov_y)) \
        -1/2*(y-mean).T @ inv_cov_y @ (y-mean)

def compute_log_post(y, theta, G, b, inv_cov_y, inv_cov_theta, mean_theta):
    k = mean_theta.size
    post_cov = G.T @ inv_cov_y @ G + inv_cov_theta
    post_mean = post_cov @ G.T @ inv_cov_y @ (y - b) + post_cov @ inv_cov_theta @ mean_theta
    return -k/2*jnp.log(2*pi) + -1/2*jnp.log(jlinalg.det(post_cov)) \
        -1/2*(theta-post_mean).T @ jlinalg.solve(post_cov, (theta-post_mean))


# FUNCTION TEMPLATES:
        


