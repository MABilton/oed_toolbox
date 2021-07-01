from math import inf
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

# Set seed for reproducability:
np.random.seed(42)

# Note that log_probs_and_grad must accept: (d, prior_samples, like_samples), and must return: (log_post, log_like_grad, log_post_grad)
def find_optimal_d(sample_likelihood, sample_prior, log_probs_and_grads, d_bounds, num_repeats=3):
    best_ape = inf
    for i in range(num_repeats):
        d, ape = minimise_ape(log_probs_and_grads, sample_prior, sample_likelihood, d_bounds)
        if ape < best_ape:
            best_d, best_ape = d, ape
    return best_d

def minimise_ape(log_probs_and_grads, sample_prior, sample_likelihood, d_bounds, num_samples=100, max_iter=100):
    # Initialise loop quantities:
    best_ape = inf
    loop_flag = True
    # Initialise stores for d and gradient:
    d = initialise_d(d_bounds)
    adagrad_s = np.zeros(d.size)
    # Initialise loop condition stores:
    ape_store, num_iter = np.empty(), 0
    # Sample from prior:
    prior_samples = sample_prior(num_samples)
    while loop_flag:
        # Draw samples from likelihood:
        like_samples = sample_likelihood(d, prior_samples)
        # Compute APE and APE gradient:
        ape, ape_grad = compute_ape_and_grad(prior_samples, like_samples, log_probs_and_grads)
        # Update Adagrad s value:
        adagrad_s += ape_grad*ape_grad
        d += adagrad_update(adagrad_s, ape_grad)
        # Take note of d value is best value observed thus far:
        if ape < best_ape:
            best_d, best_ape = d, ape
        # Store APE history and update number of iterations:
        ape_store = ape_store.append(ape) if ape_store.size else np.array(ape)
        num_iter += 1
        # Check loop conditions:
        loop_flag = check_loop_conditions(ape_store, num_iter, max_iter) if num_iter > 1 else True
    return (best_d, best_ape)

# Assume d_bounds.shape = (d_dim, 2) -> each row is a dimension of d, col 0 is LB, col 1 is UB
def initialise_d(d_bounds):
    # Uniform dist between upper and lower bounds:
    d0 = np.random.rand(d_bounds.shape[0])*(d_bounds[:,1] - d_bounds[:,0]) + d_bounds[:,0]
    return d0

def compute_ape_and_grad(d, prior_samples, like_samples, log_probs_and_grads):
    # Compute log probabilities and gradients:
    log_post, log_like_grad, log_post_grad = log_probs_and_grads(d, prior_samples, like_samples)
    # Compute APE and gradient of the APE:
    ape = jnp.mean(log_post, axis=0)
    ape_grad = log_post*log_like_grad + log_post_grad
    return (ape, ape_grad)

# See: https://machinelearningjourney.com/index.php/2021/01/05/adagrad-optimizer/
def adagrad_update(adagrad_s, grad, lr=10**-2, eps=10**-7):
    delta_d = -lr*grad/((adagrad_s + eps)**0.5)
    return delta_d

def check_loop_conditions(ape_store, num_iter, max_iter, ape_change_threshold=10**(-2)):
    # Check if number of iterations exceeds maximum number:
    if num_iter >= max_iter:
        loop_flag = False
    # Check if change in APE is below threshold:
    elif abs(ape_store[-1] - ape_store[-2]) < ape_change_threshold:
        loop_flag = False
    else:
        loop_flag = True
    return loop_flag