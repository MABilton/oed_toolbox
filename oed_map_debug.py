import numpy as np
from numpy.random import multivariate_normal as mvn
import jax
import jax.numpy as jnp

from oed_map import create_compute_map

# This script is a unit test for the create_compute_map function
# Note that for this example, we have: dim_theta = 2, dim_y = 3, dim_d = 4

def create_fun_and_map(K_fun, noise_inv, prior_mean, prior_inv):
    # Linear forward model wrt theta:
    def fun(theta, d):
        theta = jnp.atleast_1d(theta.squeeze())
        return jnp.einsum("ik,k->i", K_fun(d), theta).squeeze()
    # Analytic expression for MAP (solution to Tikhonov regularisation):
    def map(d, y):
        K = K_fun(d)
        A = K.T @ noise_inv @ K + prior_inv
        B = K.T @ noise_inv @ y + prior_inv @ prior_mean
        map = jnp.linalg.solve(A, B)
        return map
    return (fun, map)

def create_K_fun():
    # Linear coefficients of forward model (all depend on d):
    def k_fun(d):
        K = jnp.array([[d[0]**2, d[1]**(1/2)], \
                       [d[1] + 1, d[2]**(3/2)], \
                       [d[2]**(-1), d[3]**2]])
        return K
    return k_fun

if __name__ == "__main__":
    # Specify design value:
    d = np.array([1., 2., 3., 4.])
   
    # Define noise model and prior:
    noise_var = np.array([0.1, 0.1, 0.1])
    prior_var = np.array([5., 5.])
    prior_mean = np.array([1.0, -2.0])
    inv_noise_var = np.linalg.inv(np.diag(noise_var))
    inv_prior_var = np.linalg.inv(np.diag(prior_var))
    
    # Create model coefficient functions:
    K_fun = create_K_fun()

    # Create model function:
    fun, map = create_fun_and_map(K_fun, inv_noise_var, prior_mean, inv_prior_var)

    # Compute required derivatives of fun:
    fun_theta = jax.jacrev(fun, argnums=0)
    fun_d = jax.jacrev(fun, argnums=1)
    fun_2_theta = jax.jacrev(fun_theta, argnums=0)
    fun_d_theta = jax.jacrev(fun_theta, argnums=1)

    # Place vectorised functions and grads in dictionary:
    fun_dict = {}
    fun_dict["g"] = jax.vmap(fun, in_axes=(0,None))
    fun_dict["g_del_theta"] = jax.vmap(fun_theta, in_axes=(0,None))
    fun_dict["g_del_d"] = jax.vmap(fun_d, in_axes=(0,None), out_axes=0)
    fun_dict["g_del_2_theta"] = jax.vmap(fun_2_theta, in_axes=(0,None))
    fun_dict["g_del_d_theta"] = jax.vmap(fun_d_theta, in_axes=(0,None))

    # Get random samples of theta and y for a particular d:
    num_samples = 10
    prior_samples = mvn(prior_mean, np.diag(prior_var), size=num_samples)
    means = fun_dict["g"](prior_samples, d)
    like_samples = mvn(np.zeros(noise_var.size), np.diag(noise_var), size=num_samples) + means.squeeze()

    # Create function to compute MAP:
    theta_bounds = np.array([[-10., 10.], [-10., 10.]])
    compute_map = create_compute_map(fun_dict, inv_noise_var, prior_mean, inv_prior_var, theta_bounds)

    # Call function to compute MAP:
    map_val, map_grad_val = compute_map(d, like_samples)

    # Compute analytic solutions for MAP and gradient of MAP:
    map_grad = jax.jacrev(map, argnums=0)
    map_grad = jax.vmap(map_grad, in_axes=(None, 0))
    true_map_grad = map_grad(d, like_samples)
    map = jax.vmap(map, in_axes=(None, 0))
    true_map = map(d, like_samples)

    # Compare to analytic and numeric results:
    print(f"Maximum difference between analytic and true MAP value  = {np.max(abs(map_val-true_map))}")
    print(f"Maximum difference between analytic and true MAP gradient value  = {np.max(abs(map_grad_val-true_map_grad))}")