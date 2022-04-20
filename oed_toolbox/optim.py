import numpy as np
import scipy
from math import inf

def gradient_descent_for_map(lr=1e-3, abs_tol=1e-5, rel_tol=1e-5, max_iter=50, lr_step=1e-1, max_attempts=5):
    
    def gradient_descent(map_loss_and_grad, theta_0, args):
        y, d = args
        lr_i = lr
        num_attempts = 0
        optimisation_successful = False
        while not optimisation_successful:
            theta = attempt_gradient_descent(map_loss_and_grad, theta_0, y, d, lr_i)
            optimisation_successful = np.all(np.isfinite(theta)) & (num_attempts <= max_attempts)
            lr_i *= lr_step
            num_attempts += 1
        if not optimisation_successful:
            raise ValueError('Optimisation failed.')
        return theta
    
    def attempt_gradient_descent(map_loss_and_grad, theta_0, y, d, lr_i):
        num_iter = 0
        loss_prev_iter = None
        theta = theta_0
        num_opt_problems = theta_0.shape[0]
        converged = np.zeros((num_opt_problems,), dtype=bool)
        while not np.all(converged):
            loss, grad = map_loss_and_grad(theta, y, d)
            # Zero-out converged gradients:
            theta = theta - lr_i*np.einsum('a,a...->a...', ~converged, grad)
            converged = less_than_abs_tol(loss, loss_prev_iter, num_opt_problems) | \
                        less_than_rel_tol(loss, loss_prev_iter, num_opt_problems) | \
                        exceeded_max_iter(num_iter, max_iter, num_opt_problems)
            num_iter += 1
            loss_prev_iter = loss
        return theta

    def less_than_abs_tol(loss, loss_prev_iter, num_opt_problems):
        if loss_prev_iter is None:
            is_lt_abs_tol = np.zeros((num_opt_problems,), dtype=bool)
        else:
            is_lt_abs_tol = np.abs(loss - loss_prev_iter) <= abs_tol
        return is_lt_abs_tol

    def less_than_rel_tol(loss, loss_prev_iter, num_opt_problems):
        if loss_prev_iter is None:
            is_lt_rel_tol = np.zeros((num_opt_problems,), dtype=bool)
        else:
            is_lt_rel_tol = np.abs(loss - loss_prev_iter) <= rel_tol*loss_prev_iter
        return is_lt_rel_tol

    def exceeded_max_iter(num_iter, max_iter, num_opt_problems):
        return (num_iter >= max_iter)*np.ones((num_opt_problems,), dtype=bool)

    return gradient_descent

def adam_for_oed_loss(lr=1e-1, beta_1=0.9, beta_2=0.999, eps=1e-8, max_iter=100):
    def adam(oed_loss, d_0, num_samples, rng, args=None, kwargs=None, verbose=False, return_history=False):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        if return_history:
            history = {'loss': [], 'd': []}
        num_iter = 0
        best_loss, best_d = inf, None
        m_tm1, v_tm1 = 0, 0
        d = d_0
        while num_iter < max_iter:
            loss, grad = oed_loss(d, *args, num_samples=num_samples, rng=rng, **kwargs)
            if return_history:
                history['loss'].append(float(loss))
                history['d'].append(float(d))
            m_t = compute_exp_avg(new_val=grad, current_avg=m_tm1, wt=beta_1)
            v_t = compute_exp_avg(new_val=grad**2, current_avg=v_tm1, wt=beta_2)
            m_t_tilde = apply_bias_correction(m_t, num_iter, wt=beta_1)
            v_t_tilde = apply_bias_correction(v_t, num_iter, wt=beta_2)
            d -= lr*m_t_tilde/(v_t_tilde**0.5 + eps)
            if loss < best_loss:
                best_loss = loss
                best_d = d
            m_tm1, v_tm1 = m_t, v_t
            num_iter += 1
            if verbose:
                _print_optimiser_progress(num_iter, loss, d)
        return (best_d, history) if return_history else best_d
    def compute_exp_avg(new_val, current_avg, wt):
        return wt*current_avg + (1-wt)*new_val 
    def apply_bias_correction(new_avg, num_iter, wt):
        return new_avg/(1-wt**(num_iter+1))
    return adam

def _print_optimiser_progress(num_iter, loss, x):
    print(f'Iteration {num_iter}: Loss = {loss}, x = {x}')

# def scipy_minimizer_for_map(method='L-BFGS'):
#     def wrapped_minimizer(map_loss_and_grad, theta_0, y, d):

#         wrapped_loss_and_grad

#         num_opt = theta_0.shape[0]
#         theta_map = []
#         for i in range(num_opt):
#             opt_result = scipy.optimize.minimize(wrapped_loss_and_grad, theta_0[i,:], args=(y[i,:], d[i,:]), method=method, jac=True)
#             theta_map.append(opt_result['x'])
#         return np.stack(theta_map, axis=0)
#     return wrapped_minimizer