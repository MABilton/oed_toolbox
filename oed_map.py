from math import inf 
import numpy as np
from numpy import linalg
from scipy.optimize import minimize

# Set seed for reproducability:
np.random.seed(42)

# Functions to be specified by user stored in "model_funcs" dictionary:
#   - g(theta, d) = Forward model which returns (N_samples, N_theta) array
#   - g_del_theta(theta, d) = grad of g wrt theta, returning (N_samples, N_y, N_theta) array
#   - g_del_d(theta,d) = grad of g wrt d, returning (N_samples, N_y, N_d) array
#   - g_del_2_theta(theta,d) = 2nd order grad of g wrt d, returning (N_samples, N_y, N_theta, N_theta) array
#   - g_del_d_theta(theta,d) = mixed grad of g wrt d and theta, returning (N_samples, N_y, N_theta, N_d) array
# Note: theta.shape = (N_samples, N_theta), d.shape = (N_d,)

def create_compute_map(model_funcs, inv_noise, prior_mean, inv_prior, theta_bounds):
    # Unpack functions:
    g = model_funcs["g"]
    g_del_theta = model_funcs["g_del_theta"]
    g_del_d = model_funcs["g_del_d"]
    g_del_2_theta = model_funcs["g_del_2_theta"]
    g_del_d_theta = model_funcs["g_del_d_theta"]

    # Loss function to minimise to find MAP:
    def map_loss_and_grad(theta, y, d):
        # Compute loss:
        y_mean = g(theta, d)
        del_y, del_theta = (y-y_mean), (theta-prior_mean)
        loss = del_y.T @ inv_noise @ del_y + (theta-prior_mean).T @ inv_prior @ del_theta
        # Compute gradient of loss wrt theta:
        y_grad = g_del_theta(theta, d)
        loss_grad = -2*y_grad.T @ inv_noise @ del_y + 2*inv_prior @ del_theta
        return (loss, loss_grad)
        
    # Function to compute MAP and MAP Jacobian for a SINGLE sample:
    def compute_map_single_sample(d, y, num_repeats=3):
        best_loss = inf
        # Compute MAP:
        for i in range(num_repeats):
            theta_0 = np.random.rand(theta_bounds.shape[0])*(theta_bounds[:,1] - theta_bounds[:,0]) + theta_bounds[:,0]
            opt_result = minimize(map_loss_and_grad, theta_0, bounds=theta_bounds, method="L-BFGS-B", jac=True, args=(y, d))
            if opt_result["fun"] < best_loss:
                best_loss, best_theta = opt_result["fun"], opt_result["x"]
        return best_theta

    # Function to compute L11 matrix:
    def compute_L11_and_L13(theta_map, d, y_samples):
        num_samples = y_samples.shape[0]
        # Compute model values :
        g_vals = g(theta_map, d)
        G1 = g_del_theta(theta_map, d)
        G2 = g_del_d(theta_map, d)
        G11 = g_del_2_theta(theta_map, d)
        G12 = g_del_d_theta(theta_map, d)
        # Compute L11:
        G11_tilde = np.einsum("alij,alk,ak->aij", G11, inv_noise, y_samples - g_vals)
        L11_quad_term = np.einsum("ali,lk,akj->aij", G1, inv_noise, G1)
        L11 = -2*(G11_tilde - L11_quad_term - np.broadcast_to(inv_prior,(num_samples,)+inv_prior.shape))
        # Compute L13:
        G12 = g_del_d_theta(theta_map, d)
        G12_tilde = np.einsum("alij,lk,ak->aij", G12, inv_noise, y_samples - g_vals)
        L13_quad_term = np.einsum("ali,alk,akj->aij", G1, inv_noise, G2)   
        L13 = -2*(G12_tilde - L13_quad_term)
        return (L11, L13)
        
    # Function which finds MAP and Jacobian of MAP for ALL samples:
    def compute_map(d, prior_samples, like_samples):
        map_val = np.empty(), np.empty()
        # Need to manually iterate over MAP optimisations:
        for i in range(prior_samples.shape[0]):
            map_i = compute_map_single_sample(d, prior_samples[i,:], like_samples[i,:])
            map_val = np.vstack((map_val, map_i)) if map_val.size else np.array(map_val)
        # Compute gradients of MAP wrt d:
        L11, L13 = compute_L11_and_L13(map_val, d, like_samples)
        map_grad = -1*linalg.solve(L11, L13)
        return (map_val, map_grad)

    return compute_map