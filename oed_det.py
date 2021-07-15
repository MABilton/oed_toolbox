from math import inf
import numpy as np
from scipy.optimize import minimize

def find_optimal_d(post_det_and_grad, d_bounds, num_repeats=100):
    best_det = inf
    for i in range(num_repeats):
        d0 = np.random.rand(d_bounds.shape[0])*(d_bounds[:,1] - d_bounds[:,0]) + d_bounds[:,0]
        #print(abs(grad_approx - grad_mine))
        opt_result = minimize(post_det_and_grad, d0, method="L-BFGS-B", jac=True, bounds=d_bounds)
        d, det = opt_result["x"], opt_result["fun"]
        if det < best_det:
            best_d, best_det = d, det
    return best_d

def create_det_fun(model_funcs, inv_noise, inv_prior):
    # Unpack model functions:
    g_del_theta = model_funcs["g_del_theta"]
    g_del_d_theta = model_funcs["g_del_d_theta"]
    def det_and_grad(d):
        G1 = np.atleast_2d(g_del_theta(d).squeeze())
        G12 = np.atleast_3d(g_del_d_theta(d).squeeze())
        inv_cov = np.einsum("ki,kl,lj->ij", G1, inv_noise, G1) + inv_prior
        cov = np.linalg.inv(inv_cov)
        det = np.linalg.det(cov)
        inv_cov_grad = np.einsum("lik,lm,mj->ijk", G12, inv_noise, G1) + np.einsum("li,lm,mjk->ijk", G1, inv_noise, G12)
        cov_grad = -1*np.einsum("il,lmk,mj->ijk", cov, inv_cov_grad, cov)
        det_grad = det*np.einsum("ij,jik->k", inv_cov, cov_grad)
        return (np.asfortranarray(det, dtype=np.float64).squeeze(), np.asfortranarray(det_grad, dtype=np.float64))
    return det_and_grad

