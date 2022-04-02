
import numpy as np

def scipy_minimize_for_ape(scipy_minimizer, method='L-BFGS', *args, **kwargs):
    def wrapped_minimizer(map_loss_and_grad, theta_0, y, d):
        num_opt = theta_0.shape[0]
        theta_map = []
        for i in range(num_opt):
            opt_result = scipy_minimizer(map_loss_and_grad, theta_0, args=(y[i,:], d[i,:]), method=method, jac=True, *args, **kwargs)
            theta_map.append(opt_result['x'])
        return np.stack(theta_map, axis=0)
    return wrapped_minimizer

def oed_loss_for_scipy_optimiser(oed_loss):
    def wrapped_loss_and_grad(*args, **kwargs):
        loss, loss_grad = oed_loss(*args, **kwargs)
        return np.asfortranarray(loss, dtype=np.float64).squeeze(), np.asfortranarray(loss_grad, dtype=np.float64)
    return wrapped_loss_and_grad