from math import inf
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

def find_optimal_d(grad_log_like, grad_log_post, log_post, log_prior, sample_joint, d_0, optim_opts):
    best_ape = inf
    d = d_0
    for i in range(optim_opts["num_iter"]):
        joint_samples = sample_joint(optim_opts["num_samples"], d)
        ape = compute_ape(joint_samples, log_post)
        ape_grad = compute_ape_grad(joint_samples, d, grad_log_like, grad_log_post, log_post)
        d = update_d(d, ape_grad, optim_opts)
        if ape < best_ape:
            best_ape = ape
            best_d = d
    return (best_d, best_ape)

def create_det_and_grad(likelihood_cov_fun, prior_cov):
    post_det = lambda d : (jnp.atleast_2d(likelihood_cov_fun(d))@ jnp.linalg.solve((jnp.atleast_2d(likelihood_cov_fun(d)) + prior_cov), prior_cov)).squeeze()
    post_det_fun = jax.value_and_grad(post_det)
    def det_and_grad(d):
        det, grad = post_det_fun(d)
        print(det)
        return (np.array(det, dtype=np.float64), np.array(grad, dtype=np.float64))
    return det_and_grad

# Assume likelihood_cov_fun is a FUNCTION, whilst prior_cov is a CONSTANT COVARIANCE matrix:
def normal_find_optimal_d(likelihood_cov_fun, prior_cov, d_bounds, num_repeats = 9):
    best_det = inf
    post_det_fun = create_det_and_grad(likelihood_cov_fun, prior_cov)
    for i in range(num_repeats):
        d_0 = np.random.uniform(d_bounds[0][0], d_bounds[0][1], size=1)
        opt_result = minimize(post_det_fun, d_0, bounds=d_bounds, jac=True, method='L-BFGS-B')
        if opt_result['fun'] < best_det:
            best_det = opt_result['fun']
            optim_d = opt_result["x"]
            print(best_det, optim_d)
    return optim_d

def update_d(d, ape_grad, optim_opts):
    if optim_opts["algo"] == "grad_ascent":
        new_d = d + optim_opts["lr"]*ape_grad
    return new_d

def compute_ape(joint_samples, log_post):
    return jnp.mean(log_post(joint_samples["theta"], joint_samples["y"], d), axis=0)

def compute_ape_grad(samples, d, grad_log_like, grad_log_post, log_post):
    grad = log_post(joint_samples["theta"], joint_samples["y"], d)*grad_log_like(joint_samples["y"], joint_samples["theta"], d) \
    + grad_log_post(joint_samples["theta"], joint_samples["y"], d)
    return jnp.mean(grad, axis=0)