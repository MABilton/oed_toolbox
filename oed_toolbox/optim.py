import numpy as np
import scipy

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

def oed_loss_for_scipy_minimiser(oed_loss):
    def wrapped_loss_and_grad(*args, **kwargs):
        loss, loss_grad = oed_loss(*args, **kwargs)
        return np.asfortranarray(loss, dtype=np.float64).squeeze(), np.asfortranarray(loss_grad, dtype=np.float64)
    return wrapped_loss_and_grad