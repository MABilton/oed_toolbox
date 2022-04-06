from math import pi
import numpy as np
from . import utils

class Distribution:

    def _preprocess_inputs(self, **inputs):
        inputs = self._ensure_2d_shape(inputs)
        inputs = self._check_batch_dimension(inputs)
        outputs = tuple(inputs.values())
        return outputs if len(outputs) > 1 else outputs[0]

    @staticmethod
    def _ensure_2d_shape(inputs):
        for key, val in inputs.items():
            val = np.atleast_1d(val)
            if val.ndim == 1:
                inputs[key] = val[None,:]
            elif val.ndim > 2:
                raise ValueError(f'{key} must be a one or two-dimensional array.')
        return inputs

    @staticmethod
    def _check_batch_dimension(inputs):
        num_batch = np.max([val.shape[0] for val in inputs.values()])
        for key, val in inputs.items():
            if val.shape[0] == 1:
                inputs[key] = np.broadcast_to(val, shape=(num_batch, val.shape[-1]))
            elif val.shape[0] != num_batch:
                raise ValueError(f'Expected {key} input to have a batch dimension of {num_batch}; ' 
                                    f'instead, it was {val.shape[0]}.')
        return inputs

    @staticmethod
    def _reshape_logpdf_outputs(outputs, theta, d=None):
        num_samples, theta_dim = theta.shape
        if d is not None:
            d_dim = d.shape[-1]
        for key, val in outputs.items():
            if key == 'logpdf':
                outputs[key] = val.reshape(num_samples)
            elif key == 'logpdf_dt':
                outputs[key] = val.reshape(num_samples, theta_dim)
            elif key == 'logpdf_dd':
                outputs[key] = val.reshape(num_samples, d_dim)
            elif key == 'logpdf_dt_dt':
                outputs[key] = val.reshape(num_samples, theta_dim, theta_dim)
            elif key == 'logpdf_dt_dd':
                outputs[key] = val.reshape(num_samples, theta_dim, d_dim)
            elif key == 'logpdf_dt_dt_dd':
                outputs[key] = val.reshape(num_samples, theta_dim, theta_dim, d_dim)
        return outputs

    @staticmethod
    def _attempt_func_call(func, outputs_dict, args, func_key):
        if func is not None:
            outputs_dict[func_key] = func(*args)
        else:
            raise AttributeError(f'{func_key} function has not been specified.')
        return outputs_dict 

class Likelihood(Distribution):

    def __init__(self, sample=None, logpdf=None, logpdf_dt=None, logpdf_dd=None, logpdf_dt_dt=None, logpdf_dt_dd=None, logpdf_dt_dt_dd=None, logpdf_and_grads=None):
        if logpdf_and_grads is None:
            logpdf_and_grads = self._create_logpdf_and_grads(logpdf, logpdf_dt, logpdf_dd, logpdf_dt_dt, logpdf_dt_dd, logpdf_dt_dt_dd)
        self._func_dict = {'sample': sample, 'logpdf_and_grads': logpdf_and_grads}

    def sample(self, theta, d, num_samples, rng=None):
        theta, d = self._preprocess_inputs(theta=theta, d=d)
        y = self._func_dict['sample'](theta, d, num_samples, rng)
        return y.reshape(num_samples, d.shape[0], y.shape[-1])

    def logpdf(self, y, theta, d, return_logpdf=True, return_dt=False, return_dd=False, return_dt_dt=False, return_dt_dd=False, return_dt_dt_dd=False):
        theta, d, y = self._preprocess_inputs(theta=theta, d=d, y=y)
        outputs = \
        self._func_dict['logpdf_and_grads'](y, theta, d, return_logpdf, return_dt, return_dd, return_dt_dt, return_dt_dd, return_dt_dt_dd)
        return self._reshape_logpdf_outputs(outputs, theta, d)

    def _create_logpdf_and_grads(self, logpdf, logpdf_dt, logpdf_dd, logpdf_dt_dt, logpdf_dt_dd, logpdf_dt_dt_dd):
        def logpdf_and_grads(y, theta, d, return_logpdf, return_dt, return_dd, return_dt_dt, return_dt_dd, return_dt_dt_dd):
            outputs = {}
            if return_logpdf:
                outputs = self._attempt_func_call(logpdf, outputs, args=(y, theta, d), func_key='logpdf')
            if return_dt:
                outputs = self._attempt_func_call(logpdf_dt, outputs, args=(y, theta, d), func_key='logpdf_dt')
            if return_dd:
                outputs = self._attempt_func_call(logpdf_dd, outputs, args=(y, theta, d), func_key='logpdf_dd')
            if return_dt_dt:
                outputs = self._attempt_func_call(logpdf_dt_dt, outputs, args=(y, theta, d), func_key='logpdf_dt_dt')
            if return_dt_dd:
                outputs = self._attempt_func_call(logpdf_dt_dd, outputs, args=(y, theta, d), func_key='logpdf_dt_dd')
            if return_dt_dt_dd:
                outputs = self._attempt_func_call(logpdf_dt_dt_dd, outputs, args=(y, theta, d), func_key='logpdf_dt_dt_dd')         
            return outputs
        return logpdf_and_grads

    @classmethod
    def from_model_plus_constant_gaussian_noise(cls, model, noise_cov):
        
        noise_cov = np.atleast_2d(noise_cov)
        noise_icov = np.linalg.inv(noise_cov)

        def sample(theta, d, num_samples, rng):
            y_pred = model.y(theta, d)
            return utils.gaussian_sample(num_samples, mean=y_pred, cov=noise_cov, rng=rng)

        def logpdf_and_grads(y, theta, d, return_logpdf, return_dt, return_dd, return_dt_dt, return_dt_dd, return_dt_dt_dd):
            
            outputs = {}

            # Compute (shared) model evaluations:
            if return_logpdf or return_dt or return_dd or return_dt_dt or return_dt_dd or return_dt_dt_dd:
                y_pred = model.y(theta, d)
            if return_dt or return_dt_dt or return_dt_dd or return_dt_dt_dd:
                y_pred_dt = model.y_dt(theta, d)
            if return_dd or return_dt_dd or return_dt_dt_dd:
                y_pred_dd = model.y_dd(theta, d)
            if return_dt_dt or return_dt_dt_dd:
                y_pred_dt_dt = model.y_dt_dt(theta, d)
            if return_dt_dd or return_dt_dt_dd:
                y_pred_dt_dd = model.y_dt_dd(theta, d)
            
            # Compute requested outputs:
            if return_logpdf:
                outputs['logpdf'] = utils.gaussian_logpdf(y, mean=y_pred, cov=noise_cov, icov=noise_icov)
            if return_dt:
                outputs['logpdf_dt'] = 0.5*np.einsum('ij,aj,aik->ak', noise_icov + noise_icov.T, y-y_pred, y_pred_dt)
            if return_dd:
                outputs['logpdf_dd'] = 0.5*np.einsum('ij,aj,aik->ak', noise_icov + noise_icov.T, y-y_pred, y_pred_dd)
            if return_dt_dt:
                outputs['logpdf_dt_dt'] = \
                0.5*(np.einsum('aijl,ik,ak->ajl', y_pred_dt_dt, noise_icov + noise_icov.T, y-y_pred) - \
                     np.einsum('aij,ik,akl->ajl', y_pred_dt, noise_icov + noise_icov.T, y_pred_dt))
            if return_dt_dd:
                outputs['logpdf_dt_dd'] = \
                0.5*(np.einsum('aijl,ik,ak->ajl', y_pred_dt_dd, noise_icov + noise_icov.T, y-y_pred) - \
                     np.einsum('aij,ik,akl->ajl', y_pred_dt, noise_icov + noise_icov.T, y_pred_dd))
            if return_dt_dt_dd:
                y_pred_dt_dt_dd = model.y_dt_dt_dd(theta, d)
                outputs['logpdf_dt_dt_dd'] = \
                0.5*(np.einsum('aijlm,ik,ak->ajlm', y_pred_dt_dt_dd, noise_icov + noise_icov.T, y-y_pred) - \
                     np.einsum('aijl,ik,akm->ajlm', y_pred_dt_dt, noise_icov + noise_icov.T, y_pred_dd) - \
                     np.einsum('aijm,ik,akl->ajlm', y_pred_dt_dd, noise_icov + noise_icov.T, y_pred_dt) - \
                     np.einsum('aij,ik,aklm->ajlm', y_pred_dt, noise_icov + noise_icov.T, y_pred_dt_dd))

            return outputs

        return cls(sample=sample, logpdf_and_grads=logpdf_and_grads)

class Prior(Distribution):
    
    def __init__(self, sample=None, logpdf=None, logpdf_dt=None, logpdf_and_grads=None):
        if logpdf_and_grads is None:
            logpdf_and_grads = self._create_logpdf_and_grads(logpdf, logpdf_dt)
        self._func_dict = {'sample': sample, 'logpdf_and_grads': logpdf_and_grads}

    def sample(self, num_samples, rng=None):
        theta = self._func_dict['sample'](num_samples, rng).reshape(num_samples,-1)
        return theta.reshape(num_samples, 1, theta.shape[-1])

    def logpdf(self, theta, return_logpdf=True, return_dt=False):
        theta = self._preprocess_inputs(theta=theta)
        outputs =  self._func_dict['logpdf_and_grads'](theta, return_logpdf, return_dt)
        return self._reshape_logpdf_outputs(outputs, theta)

    def _create_logpdf_and_grads(self, logpdf, logpdf_dt):
        def logpdf_and_grads(theta, return_logpdf, return_dt):
            outputs = {}
            if return_logpdf:
                outputs = self._attempt_func_call(logpdf, outputs, args=(theta,), func_key='logpdf')
            if return_dt:
                outputs = self._attempt_func_call(logpdf_dt, outputs, args=(theta,), func_key='logpdf_dt')
            return outputs
        return logpdf_and_grads

    @classmethod
    def gaussian(cls, prior_mean, prior_cov):

        prior_mean = np.atleast_1d(prior_mean)
        prior_cov = np.atleast_2d(prior_cov)
        prior_icov = np.linalg.inv(prior_cov)

        def sample(num_samples, rng):
            return utils.gaussian_sample(num_samples, mean=prior_mean, cov=prior_cov, rng=rng)

        def logpdf_and_grads(theta, return_logpdf, return_dt):
            outputs = {}
            if return_logpdf:
                outputs['logpdf'] = utils.gaussian_logpdf(theta, mean=prior_mean, cov=prior_cov, icov=prior_icov)
            if return_dt:
                outputs['logpdf_dt'] = 2*np.einsum('ij,aj->ai', prior_icov, theta-prior_mean)
            return outputs

        return cls(sample=sample, logpdf_and_grads=logpdf_and_grads)

class Joint(Distribution):
    
    def __init__(self, sample=None, logpdf=None, logpdf_dt=None, logpdf_and_grads=None):
        if logpdf_and_grads is None:
            logpdf_and_grads = self._create_logpdf_and_grads(logpdf, logpdf_dt)
        self._func_dict = {'sample': sample, 'logpdf_and_grads': logpdf_and_grads}

    def sample(self, d, num_samples, rng):
        d = self._preprocess_inputs(d=d)
        theta, y = self._func_dict['sample'](d, num_samples, rng)
        return theta.reshape(num_samples, d.shape[0], theta.shape[-1]), y.reshape(num_samples, d.shape[0], y.shape[-1])

    def logpdf(self, theta, y, d, return_logpdf=True, return_dt=False):
        theta, d, y = self._preprocess_inputs(theta=theta, d=d, y=y)
        outputs = self._func_dict['logpdf_and_grads'](theta, y, d, return_logpdf, return_dt)
        return self._reshape_logpdf_outputs(outputs, theta, d)

    def _create_logpdf_and_grads(self, logpdf, logpdf_dt):
        def logpdf_and_grads(y, theta, d, return_logpdf, return_dt):
            outputs = {}
            if return_logpdf:
                outputs = self._attempt_func_call(logpdf, outputs, args=(y, theta, d), func_key='logpdf')
            if return_dt:
                outputs = self._attempt_func_call(logpdf_dt, outputs, args=(y, theta, d), func_key='logpdf_dt')
            return outputs
        return logpdf_and_grads

    @classmethod
    def from_prior_and_likelihood(cls, prior, likelihood):
        
        def sample(d, num_samples, rng):
            d = np.atleast_1d(d.squeeze())
            theta = prior.sample(num_samples, rng) # shape = (num_samples, 1, dim_theta)
            y = likelihood.sample(theta[:,0,:], d, num_samples=1, rng=rng) # shape = (1, num_samples, dim_y)
            y = np.swapaxes(y, 0, 1) # shape = (num_samples, 1, dim_y)
            return theta, y

        def logpdf_and_grads(theta, y, d, return_logpdf, return_dt):
            outputs = {}
            if return_logpdf and return_dt:
                prior_lp = prior.logpdf(theta, return_dt=True) 
                like_lp = likelihood.logpdf(y, theta, d, return_dt=True)
                outputs['logpdf'] = prior_lp['logpdf'] + like_lp['logpdf']
                outputs['logpdf_dt'] = prior_lp['logpdf_dt'] + like_lp['logpdf_dt']
            elif return_logpdf:
                outputs['logpdf'] = prior.logpdf(theta)['logpdf'] + likelihood.logpdf(y, theta, d)['logpdf']
            elif return_dt:
                outputs['logpdf_dt'] = prior.logpdf(theta, return_logpdf=False, return_dt=True)['logpdf_dt'] + likelihood.logpdf(y, theta, d, return_logpdf=False, return_dt=True)['logpdf_dt']
            return outputs

        return cls(sample=sample, logpdf_and_grads=logpdf_and_grads)

class Posterior(Distribution):

    def __init__(self, logpdf=None, logpdf_dd=None, logpdf_and_grads=None):
        if logpdf_and_grads is None:
            logpdf_and_grads = self._create_logpdf_and_grads(logpdf, logpdf_dd)
        self._func_dict = {'logpdf_and_grads': logpdf_and_grads}

    def logpdf(self, theta, y, d, return_logpdf=True, return_dd=False):
        theta, d, y = self._preprocess_inputs(theta=theta, d=d, y=y)
        outputs = self._func_dict['logpdf_and_grads'](theta, y, d, return_logpdf, return_dd)
        return self._reshape_logpdf_outputs(outputs, theta, d)

    @staticmethod
    def _create_logpdf_and_grads(logpdf, logpdf_dd):
        def logpdf_and_grads(theta, y, d, return_logpdf, return_dd):
            output = {}
            if return_logpdf:
                output = _attempt_func_call(logpdf, outputs, args=(theta, y, d), func_key='logpdf')
            if return_dd:
                output = _attempt_func_call(logpdf_dd, outputs, args=(theta, y, d), func_key='logpdf_dd')
            return output
        return logpdf_and_grads

    @classmethod
    def laplace_approximation(cls, model, minimizer, noise_cov, prior_mean, prior_cov):

        prior_mean = np.atleast_1d(prior_mean).reshape(-1)
        noise_cov = np.atleast_2d(noise_cov)
        noise_icov = np.linalg.inv(noise_cov)
        prior_cov = np.atleast_2d(prior_cov)
        prior_icov = np.linalg.inv(prior_cov)
        
        #
        #   Main Functions
        #

        def logpdf_and_grads(theta, y, d, return_logpdf, return_dd):
            outputs = {}
            if return_logpdf or return_dd:
                t_map = theta_map(theta, y, d)
                y_map = model.y(t_map, d)
                y_map_dt = model.y_dt(t_map, d)
                b = linearisation_constant(y_map, y_map_dt, t_map)
                mean, cov, icov = mean_cov_and_icov(y, t_map, y_map, y_map_dt, b)
            if return_logpdf:
                t_dim = cov.shape[0]
                outputs['logpdf'] = -0.5*(t_dim*np.log(2*pi) + np.log(np.linalg.det(cov)) + np.einsum("ai,aij,aj->a", theta-mean, icov, theta-mean))
            if return_dd:
                y_map_dd = model.y_dd(t_map, d)
                y_map_dt_dt = model.y_dt_dt(t_map, d)
                y_map_dt_dd = model.y_dt_dd(t_map, d)
                t_map_dd = theta_map_dd(y, y_map, y_map_dt, y_map_dd, y_map_dt_dt, y_map_dt_dd)
                mean_dd, cov_dd, icov_dd = mean_cov_and_icov_dd(y, y_map_dt, y_map_dd, y_map_dt_dd, t_map, t_map_dd, cov, b)
                outputs['logpdf_dd'] = -0.5*(np.einsum("aijk,aji->ak", cov_dd, icov) + \
                                             np.einsum("aijk,ai,aj->ak", icov_dd, theta-mean, theta-mean) - \
                                             2*np.einsum("alk,ali,ai->ak", mean_dd, icov, theta-mean))
            return outputs        

        #
        #   Helper Functions
        #

        def theta_map(theta_0, y, d):
            return minimizer(map_loss_and_grad, theta_0, args=(y, d))

        def map_loss_and_grad(theta, y, d):
            y_pred = model.y(theta, d)
            y_del_theta = model.y_dt(theta, d)
            # z_theta = np.einsum('ij,aj->ai', prior_icov, theta-prior_mean)
            # z_y = np.einsum('ij,aj->ai', noise_icov, y-y_pred)
            loss = np.einsum("ai,ij,aj->a", y-y_pred, noise_icov, y-y_pred) + \
                   np.einsum("ai,ij,aj->a", theta-prior_mean, prior_icov, theta-prior_mean)
            # np.einsum("ai,ai->a", theta, z_theta) + np.einsum('ai,ai->a', y-y_pred, z_y)
            loss_del_theta = -2*np.einsum("aik,ij,aj->ak", y_del_theta, noise_icov, y-y_pred) + \
                              2*np.einsum("ij,aj->ai", prior_icov, theta-prior_mean)
            # 2*z_theta - 2*np.einsum("aik,ai->ak", y_del_theta, z_y)
            return loss, loss_del_theta

        def theta_map_dd(y, y_map, y_map_dt, y_map_dd, y_map_dt_dt, y_map_dt_dd):
            map_loss_dt_dt = 2*(prior_icov - np.einsum("alij,lk,ak->aij", y_map_dt_dt, noise_icov, y-y_map) - \
                                np.einsum("ali,lk,akj->aij", y_map_dt, noise_icov, y_map_dt))
            map_loss_dt_dd = 2*(np.einsum("ali,lk,akj->aij", y_map_dt, noise_icov, y_map_dd) - \
                                np.einsum("alij,lk,ak->aij", y_map_dt_dd, noise_icov, y-y_map))
            return -1*np.linalg.solve(map_loss_dt_dt, map_loss_dt_dd)

        def linearisation_constant(y_map, y_map_del_theta, theta_map):
            return y_map - np.einsum("aij,aj->ai", y_map_del_theta, theta_map)

        def mean_cov_and_icov(y, theta_map, y_map, y_map_del_theta, b):
            inv_cov = np.einsum("aki,kl,alj->aij", y_map_del_theta, noise_icov, y_map_del_theta) + prior_icov
            cov = np.linalg.inv(inv_cov)
            mean_times_inv_cov = np.einsum("aj,jk,aki->ai", y-b, noise_icov, y_map_del_theta) + np.einsum('i,ij->j', prior_mean, prior_icov)
            mean = np.einsum("ak,aki->ai", mean_times_inv_cov, cov)
            return mean, cov, inv_cov

        def mean_cov_and_icov_dd(y, y_map_dt, y_map_dd, y_map_dt_dd, t_map, t_map_dd, cov, b):
            icov_dd = np.einsum("alik,lm,amj->aijk", y_map_dt_dd, noise_icov, y_map_dt) + \
                      np.einsum("ali,lm,amjk->aijk", y_map_dt, noise_icov, y_map_dt_dd)
            cov_dd = -1*np.einsum("ail,almk,amj->aijk", cov, icov_dd, cov)
            b_dd = np.einsum("akj,aik->aij", t_map_dd, y_map_dt) - \
                   np.einsum("aikj,ak->aij", y_map_dt_dd, t_map) - \
                   np.einsum("aik,akj->aij", y_map_dt, t_map_dd) + y_map_dd
            mean_dd = np.einsum("akij,al,lm,amk->aij", cov_dd, y-b, noise_icov, y_map_dt) -\
                      np.einsum("aki,alj,lm,amk->aij", cov, b_dd, noise_icov, y_map_dt) +\
                      np.einsum("aki,al,lm,amkj->aij", cov, y-b, noise_icov, y_map_dt_dd) +\
                      np.einsum("l,alk,akij->aij", prior_mean, cov, cov_dd)
            return mean_dd, cov_dd, icov_dd

        def approx_posterior_del_d(theta, mean, icov, mean_dd, cov_dd, icov_dd):
            return -0.5*(np.einsum("aijk,aji->ak", cov_dd, icov) + \
                         np.einsum("aijk,ai,aj->ak", icov_dd, theta-mean, theta-mean) - \
                         2*np.einsum("alk,ali,ai->ak", mean_dd, icov, theta-mean))

        return cls(logpdf_and_grads=logpdf_and_grads)

class Covariance:

    def __init__(self, cov=None, cov_dd=None, cov_and_grads=None):
        if cov_and_grads is None:
            cov_and_grads = self._create_cov_and_grads(cov, cov_dd)
        self._func_dict = {'cov_and_grads': cov_and_grads}

    @staticmethod
    def _create_cov_and_grads(cov, cov_dd):

        def cov_and_grads(d, theta_estimate, num_samples, rng, return_cov=True, return_dd=False):
            outputs = {}
            if return_cov:
                outputs = _attempt_func_call(cov, outputs, func_name='cov', args=(d, theta_estimate, num_samples, rng))
            if return_dd:
                outputs = _attempt_func_call(cov_dd, outputs, func_name='cov_dd', args=(d, theta_estimate, num_samples, rng))
            return outputs
        
        return _create_cov_and_grads

    def __call__(self, d, theta_estimate, num_samples, rng=None, return_cov=True, return_dd=True):
        return self._func_dict['cov_and_grads'](d, theta_estimate, num_samples, rng, return_cov, return_dd)

class FisherInformation(Covariance):

    def __init__(self, likelihood):
        cov_and_grad = self._create_fisher_info(likelihood)
        super().__init__(cov_and_grads=cov_and_grad)

    @staticmethod
    def _create_fisher_info(likelihood):

        def cov_and_grad(d, theta_estimate, num_samples, rng, return_cov, return_dd):
            outputs = {}
            if return_cov or return_dd:
                # Remove batch dimension:
                y = likelihood.sample(theta_estimate, d, num_samples, rng)[:,0,:]
                like_vals = likelihood.logpdf(y, theta_estimate, d, return_logpdf=False, return_dt_dt=True, return_dt_dt_dd=return_dd, return_dd=return_dd)
            if return_cov:
                outputs['cov'] = -1*like_vals['logpdf_dt_dt']
            if return_dd:
                outputs['cov_dd'] = -1*like_vals['logpdf_dt_dt_dd']
            for key, val in outputs.items():
                outputs[key] = np.mean(val, axis=0)
            return outputs

        return cov_and_grad

class PredictiveCovariance(Covariance):

    def __init__(self, model, fisher_information):
        cov_and_grad = self._create_predictive_variance(model, fisher_information)
        super().__init__(cov_and_grads=cov_and_grad)
        
    @staticmethod
    def _create_predictive_variance(model, fisher_information):
        def cov_and_grad(d, theta_estimate, num_samples, rng, return_cov, return_dd):
            outputs = {}
            if return_cov or return_dd:
                cov_and_grad = fisher_information(d, theta_estimate, num_samples, rng, return_cov=True, return_dd=return_dd)
                inv_fisher_info = np.linalg.inv(cov_and_grad['cov'])
                # Remove batch dimension:
                y_dt = model.y_dt(theta_estimate, d)[0,:]
            if return_cov:
                outputs['cov'] = np.einsum('ij,jk,kl->il', y_dt, inv_fisher_info, y_dt.T)
            if return_dd:
                # Remove batch dimension:
                y_dt_dd = model.y_dt_dd(theta_estimate, d)[0,:]
                # Derivative of inverse fisher info matrix - see Eqn (59) in Matrix cookbook (https://www2.imm.dtu.dk/pubdb/edoc/imm3274.pdf):
                fisher_info_dd = cov_and_grad['cov_dd']
                inv_fisher_info_dd = -1*np.einsum('ij,jkm,kl->ilm', inv_fisher_info, fisher_info_dd, inv_fisher_info)
                outputs['cov_dd'] = np.einsum('ijm,jk,kl->ilm', y_dt_dd, inv_fisher_info, y_dt.T) +\
                                    np.einsum('ij,jkm,kl->ilm', y_dt, inv_fisher_info_dd, y_dt.T) +\
                                    np.einsum('ij,jk,klm->ilm', y_dt, inv_fisher_info, y_dt_dd.T)
            return outputs
        return cov_and_grad