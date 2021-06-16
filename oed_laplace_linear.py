import jax
import jax.numpy as jnp
import numpy as np
from math import inf
from scipy.optimize import minimize
from jax.scipy.optimize import jminimize

def laplace_lin_find_optimal_d(likelihood_mean_fun, likelihood_cov_fun, prior_mean, prior_cov, d_bounds, num_repeats = 9):
    best_det = inf
    post_det_fun = create_det_and_grad(likelihood_mean_fun, likelihood_cov_fun, prior_mean, prior_cov)
    for i in range(num_repeats):
        d_0 = np.random.uniform(d_bounds[0][0], d_bounds[0][1], size=1)
        opt_result = minimize(post_det_fun, d_0, bounds=d_bounds, jac=True, method='L-BFGS-B')
        if opt_result['fun'] < best_det:
            best_det = opt_result['fun']
            optim_d = opt_result["x"]
            print(best_det, optim_d)
    return optim_d

def create_det_and_grad():


    def det_and_grad(d):
        # Find map for current d:
        sol = jminimize(map_loss, x0, args=(d), method="BFGS")
        theta_map, y_map = sol["x"]
        # Return approximate value of integral based on Laplace approx:
        return integral_laplace_approx(theta_map, theta_d)
    pass

# Function we want to integrate (need to find MAP wrt theta and y)
def create_map_loss(likelihood_mean, likelihood_cov, prior_mean, prior_cov):
    def map_loss(x, d):
        y, theta = x[0], x[1]
        y_mean = likelihood_mean(theta, d)
        ln_like = -0.5*(y - y_mean).T @ jnp.linalg.solve(likelihood_cov, (y - y_mean))
        ln_prior = -0.5*(theta - prior_mean).T @ jnp.linalg.solve(prior_cov, (theta - prior_mean))
        post_cov_det = jnp.linalg.det(likelihood_cov @ jnp.linalg.solve(likelihood_cov+prior_cov, prior_cov))
        n = theta.size
        return (-0.5*post_cov_det - (n/2)*jnp.ln(2*pi) + ln_like + ln_prior)*jnp.exp(ln_like)*jnp.exp(ln_prior)
    return map_loss

def create_integral_approx(likelihood_mean, likelihood_cov, prior_mean, prior_cov):
    def integral_approx(theta_map, y_map):
        
        return 
    return integral_approx

        


