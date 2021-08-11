import numpy as np
from oed_det import find_optimal_d, create_det_fun
from gp_oed_surrogate.gp.gp_create import load_gp
from gp_oed_surrogate.gp.gp_grad import create_derivative_gp

if __name__=="__main__":

    # Define noise and prior covariances:
    noise_cov, prior_cov = np.diag([0.1]), np.diag([1.])
    inv_noise, inv_prior = np.linalg.inv(noise_cov), np.linalg.inv(prior_cov)

    # Load Beam GP model:
    beam_gp = load_gp("./nonlinear_kernel_gp.json")

    # Compute GP derivatives:
    beam_theta = create_derivative_gp(beam_gp, [([0],1)])
    beam_theta_d = create_derivative_gp(beam_gp, [([0],1), ([1],1)])

    # Create model functions:
    model_funcs = {"g": beam_gp, "g_del_theta": beam_theta, "g_del_d_theta": beam_theta_d}
    det_and_grad = create_det_fun(model_funcs, inv_noise, inv_prior)

    # 
    d_bounds = np.array([[0., 180.]])
    d_opt = find_optimal_d(det_and_grad, d_bounds)
    print(d_opt)