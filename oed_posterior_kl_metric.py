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