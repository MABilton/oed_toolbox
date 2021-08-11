import numpy as np
import jax.numpy as jnp
from gp_oed_surrogate.gp.gp_create import load_gp
from gp_oed_surrogate.gp.gp_grad import create_derivative_gp
from oed_local_linear import create_local_linear_funcs
from oed_sample import create_sampling_funcs
import oed_ape
import matplotlib.pyplot as plt

np.random.seed(118)

def create_model_funcs_dict(beam_gp, beam_theta, beam_2_theta, beam_d, beam_theta_d):
    def input_wrapper(theta, d):
        output = jnp.hstack((theta, d)) 
        return output
    
    # Function which converts (theta,d) to x which can be accepted by GP:
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

    # # Code to plot slices of GP:
    # d_pts = jnp.linspace(0, 180, 1000)
    # theta_pts = jnp.linspace(1, 5, 1000)
    # d_grid, theta_grid = jnp.meshgrid(d_pts, theta_pts)
    # x = jnp.vstack((theta_grid.flatten(), d_grid.flatten())).T
    # y_pts = beam_theta_d.predict_grad_mean(x)
    # y_grid_pred = y_pts.reshape(d_pts.size,theta_pts.size)
    
    # fig, ax = plt.subplots()
    # idx = 50
    # ax.plot(d_grid[idx, :], y_grid_pred[idx, :])
    # fig.savefig("slice_d.png", dpi=300)

    # fig, ax = plt.subplots()
    # idx = 500
    # ax.plot(theta_grid[:, idx], y_grid_pred[:,idx])
    # fig.savefig("slice_theta.png", dpi=300)

    # Create dictionary of functions required for APE computations:    
    model_funcs = create_model_funcs_dict(beam_gp, beam_theta, beam_2_theta, beam_d, beam_theta_d)
    noise_cov = np.diag(np.array([0.1]))
    prior_cov = np.diag(np.array([1.0]))
    prior_mean = np.array([5.])
    theta_bounds = np.atleast_2d(np.array([1., 10.]))
    d_dim = 1
    sample_prior, sample_likelihood = create_sampling_funcs(model_funcs["g"], noise_cov, prior_mean, prior_cov)
    log_probs_and_grads = create_local_linear_funcs(model_funcs, noise_cov, prior_mean, prior_cov, theta_bounds, d_dim)
    
    # Call APE solver:
    d_bounds = np.atleast_2d(np.array([0., 180.]))
    ape_d = oed_ape.find_optimal_d(sample_likelihood, sample_prior, log_probs_and_grads, d_bounds)
    print(f"APE-Optimal d = {ape_d}")