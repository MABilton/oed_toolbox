
import pickle
from oed_posterior_kl_metric import normal_find_optimal_d

import jax
import jax.numpy as jnp
import numpy as np

from jax.scipy.linalg import cholesky, cho_solve, solve_triangular

def kernel(x_1, x_2, params):
    return params["const"]*jnp.exp(-0.5*((x_1 - x_2)/params["length"])**2)

def likelihood_cov_fun_template(surrogate):
    def likelihood_cov(d):
        (w, cov) = surrogate.predict_w_and_cov(d)
        return cov/(w**2)
    return likelihood_cov

def load_surrogate(file_name):
    with open(file_name, "rb") as f:
        gp_dict = pickle.load(f)
    return gp_dict

class LM_Surrogate:
    def __init__(self, gp_dict):
        self.K = create_kernel_matrix_func(kernel)
        self.K_diag = create_cov_diag_func(kernel)
        self.w_params = {"params": gp_dict["w_params"], "alpha": gp_dict["w_alpha"], "L": gp_dict["w_L"]}
        self.ln_diag_params = {"params": gp_dict["ln_diag_params"], "alpha": gp_dict["ln_diag_alpha"], "L": gp_dict["ln_diag_L"]}
        # self.b_params = {"params": gp_dict["b_params"], "b_alpha": gp_dict["b_alpha"], "b_L": gp_dict["b_L"]}
        self.w_train = gp_dict["w_train"]
        self.ln_diag_train = gp_dict["ln_diag_train"]
    
    def predict_w_and_cov(self, d_new):
        w = self.predict("w", d_new, self.w_params)
        ln_diag = self.predict("ln_diag", d_new, self.ln_diag_params)
        return (w, jnp.exp(ln_diag)**2)

    def predict(self, to_pred, d_new, in_params, min_var=10**(-9)):
        L, alpha, params = in_params["L"], in_params["alpha"], in_params["params"]
        x_train = self.w_train if to_pred=="w" else self.ln_diag_train
        d_new , alpha = jnp.atleast_2d(d_new), jnp.atleast_2d(alpha)
        k = self.K(x_train, d_new, params)
        alpha = alpha.reshape(k.shape)
        mean = jnp.inner(k.squeeze(), alpha.squeeze())
        #v = solve_triangular(L, k, lower=True)
        #var = self.K_diag(d_new, d_new, params) -  jnp.sum(v*v, axis=0, keepdims=True)
        #var = jax.ops.index_update(var, var<min_var, min_var)
        #return (mean.squeeze(), var.squeeze())
        return mean.squeeze()
    
def create_kernel_matrix_func(kernel_func):
    matrix_func = jax.vmap(jax.vmap(kernel_func, in_axes=(0,None,None), out_axes=0), in_axes=(None,0,None), out_axes=1)
    def matrix_scalar_func(x_1, x_2, params):
        return jnp.atleast_2d(matrix_func(x_1, x_2, params).squeeze())
    return matrix_scalar_func

def create_cov_diag_func(kernel_func):
    vectorised_kernel = jax.vmap(kernel_func, in_axes=(0,0,None), out_axes=0)
    def vectorised_kernel_func(x_1, x_2, params):
        return jnp.atleast_2d(vectorised_kernel(x_1, x_2, params).squeeze())
    return vectorised_kernel_func

def chol_decomp(A):
    try:
        L = cholesky(A, lower=True)
    except:
        L = cholesky(nearestPD(A), lower=True)
    return L

if __name__ == "__main__":

    # Load beam surrogate model into memory:
    beam_surrogate_data = load_surrogate("beam_gp.pkl")
    beam_surrogate = LM_Surrogate(beam_surrogate_data)

    # Define prior covariance matrix (just a scalar here):
    prior_cov = jnp.atleast_2d(1.)

    # Define likelihood covariance function:
    likelihood_cov_fun = likelihood_cov_fun_template(beam_surrogate)

    # Find optimal d:
    d_bounds = [(0., 180.)]
    optimal_d = normal_find_optimal_d(likelihood_cov_fun, prior_cov, d_bounds)
    print(optimal_d)