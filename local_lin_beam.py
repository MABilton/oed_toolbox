import numpy as np
import jax.numpy as jnp
from gp_oed_surrogate.gp.gp_create import load_gp
from gp_oed_surrogate.gp.gp_grad import create_derivative_gp
from oed_local_linear import create_local_linear_funcs
import oed_ape

np.random.seed(96)

def create_model_funcs_dict(beam_gp, beam_theta, beam_2_theta, beam_d, beam_theta_d):
    # Function which converts (theta,d) to x which can be accepted by GP:
    def input_wrapper(theta, d):
        x = np.hstack((theta, np.broadcast_to(d, theta.shape)))
        return x
    model_funcs = {"g": lambda theta, d: beam_gp.predict_mean(input_wrapper(theta, d)),
                   "g_del_theta": lambda theta, d: beam_theta.predict_grad_mean(input_wrapper(theta, d)),
                   "g_del_2_theta": lambda theta, d: beam_2_theta.predict_grad_mean(input_wrapper(theta, d)),
                   "g_del_d": lambda theta, d: beam_d.predict_grad_mean(input_wrapper(theta, d)),
                   "g_del_d_theta": lambda theta, d: beam_theta_d.predict_grad_mean(input_wrapper(theta, d))}
    return model_funcs

if __name__ == "__main__":
    # Load beam surrogate model:
    beam_gp = load_gp("./nonlinear_kernel_gp.json")

    # Differentiate beam_gp wrt required inputs:
    beam_theta = create_derivative_gp(beam_gp, [([0],1)])
    beam_2_theta = create_derivative_gp(beam_gp, [([0],2)])
    beam_d = create_derivative_gp(beam_gp, [([1],1)])
    beam_theta_d = create_derivative_gp(beam_gp, [([0],1), ([1],1)])

    # Create dictionary of functions required for APE computations:    
    model_funcs = create_model_funcs_dict(beam_gp, beam_theta, beam_2_theta, beam_d, beam_theta_d)
    noise_cov = np.diag(np.array([0.1]))
    prior_cov = np.diag(np.array([1.0]))
    prior_mean = np.array([5.])
    theta_bounds = np.atleast_2d(np.array([1., 10.]))
    sample_prior, sample_likelihood, log_probs_and_grads = \
        create_local_linear_funcs(model_funcs, noise_cov, prior_mean, prior_cov, theta_bounds)
    
    # Call APE solver:
    d_bounds = np.atleast_2d(np.array([80.,100.]))
    ape_d = oed_ape.find_optimal_d(sample_likelihood, sample_prior, log_probs_and_grads, d_bounds)
    print(f"APE-Optimal d = {ape_d}")
