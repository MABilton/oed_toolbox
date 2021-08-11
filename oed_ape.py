from math import inf
import numpy as np
from scipy.optimize import minimize

from oed_optimise import update_d, initialise_optim_params

# Set seed for reproducability:
np.random.seed(21)

# Note that log_probs_and_grad must accept: (d, prior_samples, like_samples), and must return: (log_post, log_like_grad, log_post_grad)
def find_optimal_d(sample_likelihood, sample_prior, log_probs_and_grads, d_bounds, optim_params=None, num_repeats=9, save_flag=True):
    best_ape = inf
    # Initialise optimisation options if not specified:
    if optim_params is None: optim_params = initialise_optim_params()
    for i in range(num_repeats):
        print(f"Optimisation {i+1}:")
        save_name = f"opt_{i}.txt" if save_flag else None
        d, ape = minimise_ape(log_probs_and_grads, sample_prior, sample_likelihood, d_bounds, optim_params, save_name)
        if ape < best_ape:
            best_d, best_ape = d, ape
    return best_d

#num_samples= 
def minimise_ape(log_probs_and_grads, sample_prior, sample_likelihood, d_bounds, optim_params, save_name, num_samples=500, max_iter=10**5, cv_beta=0.9):
    # Initialise loop quantities:
    best_ape = inf
    loop_flag = True
    # Initialise d:
    d = initialise_d(d_bounds)
    d = np.array([21.])
    # Initialise exponentially weighted average ape and ape_grad:
    d_avg, ape_avg, grad_avg = np.array([0.]), np.array([0.]), np.zeros(d.size)  
    # Initialise loop variable:
    num_iter = 0
    # Sample from prior:
    prior_samples = sample_prior(num_samples)
    if save_name is not None:
        file2write = open(save_name,'w')
    while loop_flag:
        # Compute APE and APE gradient values for all samples:
        ape, ape_grad = compute_ape_and_grads(d, prior_samples, sample_likelihood, log_probs_and_grads, ape_avg, grad_avg)
        # Store gradient:
        grad_store = np.vstack((grad_store, ape_grad)) if num_iter > 0 else np.atleast_2d(ape_grad)
        # Update number of iterations:
        num_iter += 1
        # Perform step using specified optimisation algorithm:
        d, optim_params = update_d(grad_store, optim_params, num_iter, d, d_bounds)
        # Update ape_avg and grad_avg:
        ape_avg = (cv_beta*ape + (1-cv_beta)*ape_avg)/(1-cv_beta**num_iter)
        grad_avg = (cv_beta*ape_grad + (1-cv_beta)*grad_avg)/(1-cv_beta**num_iter)
        d_avg = (cv_beta*d + (1-cv_beta)*d_avg)/(1-cv_beta**num_iter)
        print(f"{ape}, {d}")
        if save_name is not None:
            file2write.write(f"{d.item()}, {ape}, {ape_grad}\n")
        # Take note of whether d value is best value observed thus far:
        if ape < best_ape:
            best_d, best_ape = d, ape
        # Store APE and d history:
        ape_store = np.vstack((ape_store, ape)) if num_iter>1 else np.atleast_1d(ape)
        d_store = np.vstack((d_store, d)) if num_iter>1 else np.atleast_2d(d)
        # Check loop conditions:
        loop_flag = check_loop_conditions(d, d_avg, d_bounds, num_iter, max_iter) if num_iter > 1 else True
    if save_name is not None:
        file2write.close()
    return (best_d, best_ape)

# Assume d_bounds.shape = (d_dim, 2) -> each row is a dimension of d, col 0 is LB, col 1 is UB
def initialise_d(d_bounds):
    # Uniform dist between upper and lower bounds:
    d0 = np.random.rand(d_bounds.shape[0])*(d_bounds[:,1] - d_bounds[:,0]) + d_bounds[:,0]
    return d0

def compute_ape_and_grads(d, prior_samples, sample_likelihood, log_probs_and_grads, ape_avg, grad_avg, cv_flag=True, rb_num=0):
    # Repeat d:
    d_dim, num_samples = d.size, prior_samples.shape[0]
    d_repeat = np.repeat(d, num_samples, axis=0).reshape(num_samples,d_dim)
    if not rb_num:
        ape, ape_grad = non_rb_ape(d_repeat, prior_samples, sample_likelihood, log_probs_and_grads, cv_flag, ape_avg, grad_avg)
    else: 
        ape, ape_grad = rb_ape(d_repeat, prior_samples, sample_likelihood, log_probs_and_grads, cv_flag, rb_num, ape_avg, grad_avg)
    return (ape, ape_grad)

def non_rb_ape(d, prior_samples, sample_likelihood, log_probs_and_grads, cv_flag, ape_avg, grad_avg):
    # Sample likelihood:
    like_samples = sample_likelihood(d, prior_samples)
    # Compute log probabilities and gradients:
    log_post, log_like_grad, log_post_grad = log_probs_and_grads(d, prior_samples, like_samples)
    # Compute ape_grad term for each sample:
    grad = np.einsum("a,ai->ai", log_post, log_like_grad) + log_post_grad
    # Apply control variates to compute ape and ape_grad if requested:
    if cv_flag:
        ape, ape_grad = apply_control_variates(log_post, grad, log_like_grad, ape_avg, grad_avg)
    else:
        ape, ape_grad = -1*np.mean(log_post, axis=0), -1*np.mean(grad, axis=0)
    return (ape, ape_grad)

def rb_ape(d, prior_samples, sample_likelihood, log_probs_and_grads, cv_flag, inner_samples, ape_avg, grad_avg):
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
    log_like_grad_y_avg =  np.mean(log_like_grad.reshape(outer_samples, inner_samples, grad.shape[1]), axis=1)
    # Compute APE and APE grad by averaging these expectations over theta:
    if cv_flag:
        ape, ape_grad = apply_control_variates(log_post_y_avg, grad_y_avg, log_like_grad_y_avg, ape_avg, grad_avg)
    else:
        ape, ape_grad = -1*np.mean(log_post_y_avg, axis=0), -1*np.mean(grad_y_avg, axis=0)
    return (ape, ape_grad)

def apply_control_variates(log_post, grad, log_like_grad, ape_avg, grad_avg):
    # Compute control variates to decrease variance of ape and apge_grad estimates:
    if False: #  # abs(ape_avg) > 1 and abs(grad_avg) > 1
        cv_grad = np.einsum("ai,j->aij", log_like_grad, grad_avg).reshape(grad.shape[0], grad.shape[1]**2)
        cv = np.hstack((log_like_grad, ape_avg*log_like_grad, cv_grad))
    else:
        cv = log_like_grad
    num_cv = cv.shape[1]
    del_cv = cv # (cv - np.mean(cv, axis=0))
    var_cv = np.mean(np.einsum("ai,aj->aij", del_cv, del_cv), axis=0)
    cov_ape = np.mean(np.einsum("a,ai->ai", (log_post-np.mean(log_post, axis=0)), del_cv), axis=0)
    cov_grad = np.mean(np.einsum("aj,ai->aij",(grad-np.mean(grad, axis=0)), del_cv), axis=0)
    a_ape = np.linalg.solve(var_cv, cov_ape)
    a_grad = np.linalg.solve(var_cv, cov_grad).reshape(num_cv, grad.shape[1])
    # Compute APE and gradient of the APE using control variates:
    ape = -1*np.mean(log_post - np.einsum("i,ai->a", a_ape, cv), axis=0)
    ape_grad = -1*np.mean(grad - np.einsum("ij,ai->aj", a_grad, cv), axis=0)
    return (ape, ape_grad)

def check_loop_conditions(d, d_avg, d_bounds, num_iter, max_iter, d_threshold=1*10**-9):
    # Check change in APE and d vs mean:
    d_change_flag = np.all(abs(d - d_avg) < d_threshold)
    # Check if d currently at boundary:
    lb, ub = d_bounds[:,0], d_bounds[:,1]
    boundary_flag = np.all((abs(d - lb) < 10**-3) | (abs(d - ub) < 10**-3))
    # Check if number of iterations exceeds maximum number:
    if num_iter >= max_iter:
        loop_flag = False
    # Check if change in APE AND d is below threshold:
    elif d_change_flag or boundary_flag:
        loop_flag = False
    else:
        loop_flag = True
    return loop_flag