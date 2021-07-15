import jax
import jax.numpy as jnp
import numpy as np
from math import inf
from scipy.optimize import minimize

def create_det_and_grad(likelihood_cov_fun, prior_cov):
    post_det = lambda d : (jnp.atleast_2d(likelihood_cov_fun(d))@ jnp.linalg.solve((jnp.atleast_2d(likelihood_cov_fun(d)) + prior_cov), prior_cov)).squeeze()
    post_det_fun = jax.value_and_grad(post_det)
    def det_and_grad(d):
        det, grad = post_det_fun(d)
        print(det)
        return (np.array(det, dtype=np.float64), np.array(grad, dtype=np.float64))
    return det_and_grad

# Assume likelihood_cov_fun is a FUNCTION, whilst prior_cov is a CONSTANT COVARIANCE matrix:
def global_lin_find_optimal_d(likelihood_cov_fun, prior_cov, d_bounds, num_repeats = 9):
    best_det = inf
    post_det_fun = create_det_and_grad(likelihood_cov_fun, prior_cov)
    for i in range(num_repeats):
        d_0 = np.random.uniform(d_bounds[0][0], d_bounds[0][1], size=1)
        opt_result = minimize(post_det_fun, d_0, bounds=d_bounds, jac=True, method='L-BFGS-B')
        if opt_result['fun'] < best_det:
            best_det = opt_result['fun']
            optim_d = opt_result["x"]
            print(best_det, optim_d)
    return optim_d