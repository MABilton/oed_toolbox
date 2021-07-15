from math import inf
import numpy as np
from scipy.optimize import minimize

from oed_optimise import update_d, initialise_optim_params

# Set seed for reproducability:
np.random.seed(42)

# Note that log_probs_and_grad must accept: (d, prior_samples, like_samples), and must return: (log_post, log_like_grad, log_post_grad)
def find_optimal_d(sample_likelihood, sample_prior, log_probs_and_grads, d_bounds, optim_params=None, num_repeats=9):
    best_ape = inf
    # Initialise optimisation options if not specified:
    if optim_params is None: optim_params = initialise_optim_params()
    for i in range(num_repeats):
        print(f"Optimisation {i+1}:")
        d, ape = minimise_ape(log_probs_and_grads, sample_prior, sample_likelihood, d_bounds, optim_params)
        if ape < best_ape:
            best_d, best_ape = d, ape
    return best_d

#num_samples= 
def minimise_ape(log_probs_and_grads, sample_prior, sample_likelihood, d_bounds, optim_params, num_samples=10, max_iter=10**3):
    # Initialise loop quantities:
    best_ape = inf
    loop_flag = True
    # Initialise d:
    d = initialise_d(d_bounds)
    # Initialise loop condition stores:
    ape_store, d_store, num_iter = [], [], 0
    # Sample from prior:
    prior_samples = sample_prior(num_samples)
    while loop_flag:
        # Compute APE and APE gradient values for all samples:
        ape, ape_grad = compute_ape_and_grads(d, prior_samples, sample_likelihood, log_probs_and_grads)
        # Store gradient:
        grad_store = np.vstack((grad_store, ape_grad)) if num_iter > 0 else np.atleast_2d(ape_grad)
        num_iter += 1
        # Perform step using specified optimisation algorithm:
        update, optim_params = update_d(grad_store, optim_params, num_iter)
        d -= update
        print(f'{ape}, {d}')
        # Take note of d value is best value observed thus far:
        if ape < best_ape:
            best_d, best_ape = d, ape
        # Store APE and d history and update number of iterations:
        ape_store.append(ape)
        d_store.append(d)
        # Check loop conditions:
        loop_flag = check_loop_conditions(ape_store, d_store, num_iter, max_iter) if num_iter > 1 else True
    return (best_d, best_ape)

# Assume d_bounds.shape = (d_dim, 2) -> each row is a dimension of d, col 0 is LB, col 1 is UB
def initialise_d(d_bounds):
    # Uniform dist between upper and lower bounds:
    d0 = np.random.rand(d_bounds.shape[0])*(d_bounds[:,1] - d_bounds[:,0]) + d_bounds[:,0]
    return d0

def compute_ape_and_grads(d, prior_samples, sample_likelihood, log_probs_and_grads, cv_flag=True, rb_num=None):
    if not rb_num:
        ape, ape_grad = non_rb_ape(d, prior_samples, sample_likelihood, log_probs_and_grads, cv_flag)
    else: 
        ape, ape_grad = rb_ape(d, prior_samples, sample_likelihood, log_probs_and_grads, cv_flag, rb_num)
    return (ape, ape_grad)

def non_rb_ape(d, prior_samples, sample_likelihood, log_probs_and_grads, cv_flag):
    # Sample likelihood:
    like_samples = sample_likelihood(d, prior_samples)
    # Compute log probabilities and gradients:
    log_post, log_like_grad, log_post_grad = log_probs_and_grads(d, prior_samples, like_samples)
    # Compute ape_grad term for each sample:
    grad = np.einsum("a,ai->ai", log_post, log_like_grad) + log_post_grad
    # Apply control variates to compute ape and ape_grad if requested:
    if cv_flag:
        ape, ape_grad = apply_control_variates(log_post, grad, log_like_grad, log_post_grad)
    else:
        ape, ape_grad = -1*np.mean(log_post, axis=0), -1*np.mean(grad, axis=0)
    return (ape, ape_grad)

def rb_ape(d, prior_samples, sample_likelihood, log_probs_and_grads, cv_flag, inner_samples):
    outer_samples = prior_samples.shape[0]
    # For each theta we've sampled, sample inner_samples from likelihood:
    prior_repeat = np.repeat(prior_samples, inner_samples, axis=0)
    like_samples = sample_likelihood(d, prior_repeat)
    # Compute log probabilities and gradients:
    log_post, log_like_grad, log_post_grad = log_probs_and_grads(d, prior_repeat, like_samples)
    # Compute gradient for each sample:
    grad = np.einsum("a,ai->ai", log_post, log_like_grad) + log_post_grad
    # Compute expectations taken wrt p(y|theta,d):
    log_post_y_avg = np.mean(log_post.reshape(outer_samples, inner_samples), axis=1)
    grad_y_avg  = np.mean(grad.reshape(outer_samples, inner_samples, grad.shape[1]), axis=1)
    log_post_grad_y_avg = np.mean(log_post_grad.reshape(outer_samples, inner_samples, grad.shape[1]), axis=1)
    log_like_grad_y_avg =  np.mean(log_like_grad.reshape(outer_samples, inner_samples, grad.shape[1]), axis=1)
    # Compute APE and APE grad by averaging these expectations over theta:
    if cv_flag:
        ape, ape_grad = apply_control_variates(log_post_y_avg, grad_y_avg, log_like_grad_y_avg, log_post_grad_y_avg)
    else:
        ape, ape_grad = -1*np.mean(log_post_y_avg, axis=0), -1*np.mean(grad_y_avg, axis=0)
    return (ape, ape_grad)

def apply_control_variates(log_post, grad, log_like_grad, log_post_grad):
    # Compute control variates to decrease variance of ape and apge_grad estimates:
    cv = np.hstack((log_like_grad, log_post_grad))
    del_cv = (cv - np.mean(cv, axis=0))
    var_cv = np.mean(np.einsum("ai,aj->aij", del_cv, del_cv), axis=0)
    cov_ape = np.mean(np.einsum("a,ai->ai", (log_post-np.mean(log_post, axis=0)), del_cv), axis=0).squeeze()
    cov_grad = np.mean(np.einsum("ai,aj->aij",(grad-np.mean(grad, axis=0)), del_cv), axis=0).squeeze()
    a_ape = np.atleast_1d(np.linalg.solve(var_cv, cov_ape).squeeze())
    a_grad = np.atleast_2d(np.linalg.solve(var_cv, cov_grad).squeeze())
    # Compute APE and gradient of the APE using control variates:
    ape = -1*np.mean(log_post - np.einsum("i,ai->a", a_ape, cv), axis=0)
    ape_grad = -1*np.mean(grad - np.einsum("ij,aj->ai", a_grad, cv), axis=0)
    return (ape, ape_grad)

def check_loop_conditions(ape_store, d_store, num_iter, max_iter, ape_change_threshold=1*10**-4, d_change_threshold=10**-3):
    ape_change_flag = abs(ape_store[-1] - ape_store[-2]) < ape_change_threshold
    d_change_flag = abs(d_store[-1] - d_store[-2]) < d_change_threshold
    # Check if number of iterations exceeds maximum number:
    if num_iter >= max_iter:
        loop_flag = False
    # Check if change in APE AND d is below threshold:
    elif ape_change_flag and d_change_flag:
        loop_flag = False
    else:
        loop_flag = True
    return loop_flag