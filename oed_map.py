from math import inf 
import numpy as np
from numpy import linalg
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Set seed for reproducability:
np.random.seed(2)

# Functions to be specified by user stored in "model_funcs" dictionary:
#   - g(theta, d) = Forward model which returns (N_samples, N_theta) array
#   - g_del_theta(theta, d) = grad of g wrt theta, returning (N_samples, N_y, N_theta) array
#   - g_del_d(theta, d) = grad of g wrt d, returning (N_samples, N_y, N_d) array
#   - g_del_2_theta(theta, d) = 2nd order grad of g wrt d, returning (N_samples, N_y, N_theta, N_theta) array
#   - g_del_d_theta(theta, d) = mixed grad of g wrt d and theta, returning (N_samples, N_y, N_theta, N_d) array
# Note: theta.shape = (N_samples, N_theta), d.shape = (N_d,)

def create_compute_map(model_funcs, inv_noise, prior_mean, inv_prior, theta_bounds, dim_d):
    # Unpack functions:
    g = model_funcs["g"]
    g_del_theta = model_funcs["g_del_theta"]
    g_del_d = model_funcs["g_del_d"]
    g_del_2_theta = model_funcs["g_del_2_theta"]
    g_del_d_theta = model_funcs["g_del_d_theta"]

    # Get dimension of theta:
    dim_theta = prior_mean.size

    # Loss function to minimise to find MAP:
    def map_loss_and_grad(theta, y, d):
        theta, d = theta.reshape(1,dim_theta), d.reshape(1, dim_d)
        # Compute loss:
        y_mean = g(theta, d)
        # Transpose so both are column vectors:
        del_y, del_theta = np.atleast_1d((y-y_mean).squeeze()), np.atleast_1d((theta-prior_mean).squeeze())
        loss = np.einsum("i,ij,j->", del_y, inv_noise, del_y) + np.einsum("i,ij,j->", del_theta, inv_prior, del_theta)
        # Compute gradient of loss wrt theta - squeeze to remove singleton dimension:
        y_grad = np.atleast_2d(g_del_theta(theta, d).squeeze())
        loss_grad = -2*np.einsum("ik,ij,j->k", y_grad, inv_noise, del_y) + 2*np.einsum("ij,j->i", inv_prior, del_theta)
        return (np.asfortranarray(loss, dtype=np.float64).squeeze(), np.asfortranarray(loss_grad, dtype=np.float64))
    
    # Function to initialise theta:
    def initialise_theta(theta_bounds, theta_init):
        if theta_init is None:
            theta_0 = np.random.rand(theta_bounds.shape[0])*(theta_bounds[:,1] - theta_bounds[:,0]) + theta_bounds[:,0]
        else:
            theta_0 = theta_init
        return theta_0

    # Function to compute MAP and MAP Jacobian for a SINGLE sample:
    def compute_map_single_sample(d, y, theta_init, theta_samp, num_repeats=1):
        best_loss = inf
        # Compute MAP:
        loss_only = lambda theta, y, d :  map_loss_and_grad(theta, y, d)[0]

        for i in range(num_repeats):
            theta_0 = initialise_theta(theta_bounds, theta_init)
            opt_result = minimize(map_loss_and_grad, theta_0, bounds=theta_bounds, jac=True, args=(y, d)) #  method="L-BFGS-B",
            # opt_result = minimize(loss_only, theta_0, bounds=theta_bounds, jac=False, args=(y, d)) # method="L-BFGS-B", 
            if opt_result["fun"] < best_loss:
                best_loss, best_theta = opt_result["fun"], opt_result["x"]
                best_theta_0 = theta_0
        
        # # Plot loss function + MAP:
        # loss_vals, theta_vals = [], []
        # for theta in np.linspace(1,10,200):
        #     print(theta)
        #     loss_vals.append(loss_only(theta,y,d))
        #     theta_vals.append(theta)
        # plt.plot(np.array(theta_vals), np.array(loss_vals), label=f"theta_sample = {theta_samp.item():.4f}, d = {d.item():.2f}, y = {y.item():.2f}")
        # plt.plot(best_theta, best_loss, marker="o", label=f"Map = {best_theta.item():.4f}, theta_0 = {best_theta_0.item():.4f}")
        # plt.xticks(np.arange(0,10.5,0.5))
        # plt.xlabel("Theta")
        # plt.ylabel('Loss')
        # plt.legend()
        # # plt.ylim(0, 100)
        # plt.savefig("Loss Values.png")

        return best_theta

    # Function to compute L11 matrix:
    def compute_L11_and_L13(theta_map, d, y_samples):
        # Compute model values:
        g_vals = g(theta_map, d)
        G1 = g_del_theta(theta_map, d)
        G2 = g_del_d(theta_map, d)
        G11 = g_del_2_theta(theta_map, d)
        G12 = g_del_d_theta(theta_map, d)
        # Compute L11:
        G11_tilde = np.einsum("alij,lk,ak->aij", G11, inv_noise, y_samples - g_vals)
        L11_quad_term = np.einsum("ali,lk,akj->aij", G1, inv_noise, G1)
        L11 = -2*(G11_tilde - L11_quad_term - inv_prior)
        # Compute L13:
        G12 = g_del_d_theta(theta_map, d)
        G12_tilde = np.einsum("alij,lk,ak->aij", G12, inv_noise, y_samples - g_vals)
        L13_quad_term = np.einsum("ali,lk,akj->aij", G1, inv_noise, G2)   
        L13 = -2*(G12_tilde - L13_quad_term)
        return (L11, L13)
        
    # Function which finds MAP and Jacobian of MAP for ALL samples:
    def compute_map(d, like_samples, theta_samp, theta_init=None):
        num_samples = like_samples.shape[0]
        map_val = []
        # Need to manually iterate over MAP optimisations:
        if theta_init is None:
            for i in range(num_samples):
                theta_init_i = theta_init if theta_init is None else theta_init[i,:]
                map_i = compute_map_single_sample(d[i,:], like_samples[i,:], theta_init_i, theta_samp[i,:])
                map_val.append(map_i)
            map_val = np.array(map_val).reshape(num_samples, dim_theta)
        else:
            map_val = theta_init
        # Compute gradients of MAP wrt d:
        L11, L13 = compute_L11_and_L13(map_val, d, like_samples)
        map_grad = -1*linalg.solve(L11, L13)
        return (map_val, map_grad)
    return compute_map