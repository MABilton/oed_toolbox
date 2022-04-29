from math import pi
import jax
import numpy as np
from . import utils

class Distribution:

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
        return outputs

class Likelihood(Distribution):

    def __init__(self, sample=None, logpdf=None, logpdf_dy=None, logpdf_dt=None, logpdf_dd=None, logpdf_dt_dt=None, logpdf_dt_dd=None, logpdf_dt_dy=None, logpdf_and_grads=None, sample_base=None, transform=None, transform_dd=None, transform_and_grads=None):
        if logpdf_and_grads is None:
            logpdf_and_grads = \
            self._create_logpdf_and_grads(logpdf, logpdf_dy, logpdf_dt, logpdf_dd, logpdf_dt_dt, logpdf_dt_dd, logpdf_dt_dy)
        if transform_and_grads is None:
            transform_and_grads = self._create_transform_and_grads(transform, transform_dd)
        self._func_dict = {'sample': sample, 'sample_base': sample_base, 'logpdf_and_grads': logpdf_and_grads, 'transform_and_grads': transform_and_grads}

    #
    #   Sampling and Probability Methods
    #

    def sample(self, theta, d, num_samples, rng=None):
        if 'sample' not in self._func_dict:
            return AttributeError('Sampling function not specified.')
        theta, d = utils._preprocess_inputs(theta=theta, d=d)
        self._check_sample_dimension(num_samples, theta, d) 
        y = self._func_dict['sample'](theta, d, num_samples, rng)
        return y.reshape(num_samples, y.shape[-1])

    @staticmethod
    def _check_sample_dimension(num_samples, theta, d):
        for val, name in zip([theta, d], ['theta', 'd']):
            if (val.shape[0] != num_samples) and (val.shape[0] != 1):
                raise ValueError(f'Expected {name}.shape = (num_samples, {val.shape[-1]}) ' 
                                 f'= ({num_samples}, {val.shape[-1]}); instead, {name}.shape = ({val.shape})')

    def sample_base(self, num_samples, rng=None):
        if 'sample_base' not in self._func_dict:
            return AttributeError('Base distribution sampling function not specified.')
        epsilon = self._func_dict['sample_base'](num_samples, rng)
        return epsilon.reshape(num_samples, epsilon.shape[-1])

    def logpdf(self, y, theta, d, return_logpdf=True, return_dy=True, return_dt=False, return_dd=False, return_dt_dt=False, return_dt_dd=False, return_dt_dy=False):
        theta, d, y = utils._preprocess_inputs(theta=theta, d=d, y=y)
        outputs = \
        self._func_dict['logpdf_and_grads'](y, theta, d, return_logpdf, return_dy, return_dt, return_dd, return_dt_dt, return_dt_dd, return_dt_dy)
        return self._reshape_logpdf_outputs(outputs, theta, d)

    def transform(self, epsilon, theta, d, return_dd=False):
        epsilon, theta, d = utils._preprocess_inputs(epsilon=epsilon, theta=theta, d=d)
        outputs = self._func_dict['transform_and_grads'](epsilon, theta, d, return_dd)
        return self._reshape_transform_outputs(outputs, d)

    @staticmethod
    def _reshape_transform_outputs(outputs, d):
        num_batch, d_dim = d.shape
        for key, val in outputs.items():
            y_dim = val.shape[-1]
            if key == 'y':
                outputs[key] = val.reshape(num_batch, y_dim)
            elif key == 'y_dd':
                outputs[key] = val.reshape(num_batch, y_dim, d_dim)
        return outputs

    #
    #   Constructor Methods
    #

    @classmethod
    def from_model_plus_constant_gaussian_noise(cls, model, noise_cov):
         
        noise_cov = np.atleast_2d(noise_cov)
        noise_chol = np.linalg.cholesky(noise_cov)
        noise_icov = np.linalg.inv(noise_cov)
        y_dim = noise_cov.shape[0]

        def sample_base(num_samples, rng):
            return utils.unit_gaussian_sample(y_dim, num_samples, rng)

        def transform_and_grads(epsilon, theta, d, return_dd):
            outputs = {'y': utils.gaussian_transform(epsilon, mean=model.predict(theta,d), cov_chol=noise_chol)}
            if return_dd:
                outputs['y_dd'] = model.predict_dd(theta, d)
            return outputs

        def sample(theta, d, num_samples, rng):
            epsilon = sample_base(num_samples, rng=rng)
            return transform_and_grads(epsilon, theta, d, return_dd=False)['y'] 

        def logpdf_and_grads(y, theta, d, return_logpdf, return_dy, return_dt, return_dd, return_dt_dt, return_dt_dd, return_dt_dy):
            outputs = {}
            # Compute (shared) model evaluations:
            if return_logpdf or return_dy or return_dt or return_dd or return_dt_dt or return_dt_dd or return_dt_dy:
                y_pred = model.predict(theta, d)
            if return_dt or return_dt_dt or return_dt_dd or return_dt_dy:
                y_pred_dt = model.predict_dt(theta, d)
            if return_dd or return_dt_dd:
                y_pred_dd = model.predict_dd(theta, d)
            if return_dt_dt:
                y_pred_dt_dt = model.predict_dt_dt(theta, d)
            if return_dt_dd:
                y_pred_dt_dd = model.predict_dt_dd(theta, d)
            # Compute requested outputs:
            if return_logpdf:
                outputs['logpdf'] = utils.gaussian_logpdf(y, mean=y_pred, cov=noise_cov, icov=noise_icov)
            if return_dy:
                outputs['logpdf_dy'] = -0.5*np.einsum('ij,aj->ai', noise_icov + noise_icov.T, y-y_pred)
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
            if return_dt_dy:
                outputs['logpdf_dt_dy'] = 0.5*np.einsum('im,aik->akm', noise_icov + noise_icov.T, y_pred_dt)
            return outputs

        return cls(sample=sample, sample_base=sample_base, logpdf_and_grads=logpdf_and_grads, transform_and_grads=transform_and_grads)

    def _create_logpdf_and_grads(self, logpdf, logpdf_dy, logpdf_dt, logpdf_dd, logpdf_dt_dt, logpdf_dt_dd, logpdf_dt_dt_dd):
        def logpdf_and_grads(y, theta, d, return_logpdf, return_dy, return_dt, return_dd, return_dt_dt, return_dt_dd, return_dt_dy):
            outputs = {}
            if return_logpdf:
                outputs = utils._attempt_func_call(logpdf, outputs, args=(y, theta, d), func_key='logpdf')
            if return_dy:
                outputs = utils._attempt_func_call(logpdf_dy, outputs, args=(y, theta, d), func_key='logpdf_dy')
            if return_dt:
                outputs = utils._attempt_func_call(logpdf_dt, outputs, args=(y, theta, d), func_key='logpdf_dt')
            if return_dd:
                outputs = utils._attempt_func_call(logpdf_dd, outputs, args=(y, theta, d), func_key='logpdf_dd')
            if return_dt_dt:
                outputs = utils._attempt_func_call(logpdf_dt_dt, outputs, args=(y, theta, d), func_key='logpdf_dt_dt')
            if return_dt_dd:
                outputs = utils._attempt_func_call(logpdf_dt_dd, outputs, args=(y, theta, d), func_key='logpdf_dt_dd')
            if return_dt_dy:
                outputs = utils._attempt_func_call(return_dt_dy, outputs, args=(y, theta, d), func_key='return_dt_dy')         
            return outputs
        return logpdf_and_grads

    def _create_transform_and_grads(self, transform, transform_dd):
        def transform_and_grads(theta, d, return_dd):
            outputs = {}
            outputs = utils._attempt_func_call(transform, outputs, args=(theta, d), func_key='y')
            if return_dd:
                outputs = utils._attempt_func_call(transform_dd, outputs, args=(theta, d), func_key='y_dd')
            return outputs
        return transform_and_grads

class Prior(Distribution):
    
    def __init__(self, sample=None, logpdf=None, logpdf_dt=None, logpdf_and_grads=None):
        if logpdf_and_grads is None:
            logpdf_and_grads = self._create_logpdf_and_grads(logpdf, logpdf_dt)
        self._func_dict = {'sample': sample, 'logpdf_and_grads': logpdf_and_grads}

    def sample(self, num_samples, rng=None):
        theta = self._func_dict['sample'](num_samples, rng).reshape(num_samples,-1)
        return theta.reshape(num_samples, theta.shape[-1])

    def logpdf(self, theta, return_logpdf=True, return_dt=False):
        theta = utils._preprocess_inputs(theta=theta)
        outputs =  self._func_dict['logpdf_and_grads'](theta, return_logpdf, return_dt)
        return self._reshape_logpdf_outputs(outputs, theta)

    def _create_logpdf_and_grads(self, logpdf, logpdf_dt):
        def logpdf_and_grads(theta, return_logpdf, return_dt):
            outputs = {}
            if return_logpdf:
                outputs = utils._attempt_func_call(logpdf, outputs, args=(theta,), func_key='logpdf')
            if return_dt:
                outputs = utils._attempt_func_call(logpdf_dt, outputs, args=(theta,), func_key='logpdf_dt')
            return outputs
        return logpdf_and_grads

    @classmethod
    def gaussian(cls, prior_mean, prior_cov):

        prior_mean = np.atleast_1d(prior_mean)
        prior_cov = np.atleast_2d(prior_cov)
        prior_chol = np.linalg.cholesky(prior_cov)
        prior_icov = np.linalg.inv(prior_cov)

        def sample(num_samples, rng):
            return utils.gaussian_sample(num_samples, mean=prior_mean, cov_chol=prior_chol, rng=rng)

        def logpdf_and_grads(theta, return_logpdf, return_dt):
            outputs = {}
            if return_logpdf:
                outputs['logpdf'] = utils.gaussian_logpdf(theta, mean=prior_mean, cov=prior_cov, icov=prior_icov)
            if return_dt:
                outputs['logpdf_dt'] = 2*np.einsum('ij,aj->ai', prior_icov, theta-prior_mean)
            return outputs

        return cls(sample=sample, logpdf_and_grads=logpdf_and_grads)

class Posterior(Distribution):

    def __init__(self, logpdf=None, logpdf_dd=None, logpdf_dy=None, logpdf_and_grads=None):
        if logpdf_and_grads is None:
            logpdf_and_grads = self._create_logpdf_and_grads(logpdf, logpdf_dd, logpdf_dy)
        self._func_dict = {'logpdf_and_grads': logpdf_and_grads}

    def logpdf(self, theta, y, d, return_logpdf=True, return_dd=False, return_dy=False):
        theta, d, y = utils._preprocess_inputs(theta=theta, d=d, y=y)
        outputs = self._func_dict['logpdf_and_grads'](theta, y, d, return_logpdf, return_dd, return_dy)
        return self._reshape_logpdf_outputs(outputs, theta, d)

    @staticmethod
    def _create_logpdf_and_grads(logpdf, logpdf_dd, logpdf_dy):
        def logpdf_and_grads(theta, y, d, return_logpdf, return_dd, return_dy):
            outputs = {}
            if return_logpdf:
                outputs = utils._attempt_func_call(logpdf, outputs, args=(theta, y, d), func_key='logpdf')
            if return_dd:
                outputs = utils._attempt_func_call(logpdf_dd, outputs, args=(theta, y, d), func_key='logpdf_dd')
            if return_dy:
                outputs = utils._attempt_func_call(logpdf_dy, outputs, args=(theta, y, d), func_key='logpdf_dy')
            return outputs
        return logpdf_and_grads

    @classmethod
    def from_approx_post(cls, approx_post):
        logpdf = lambda theta, y, d : approx_post.logpdf(theta[:,None,:], x=y, d=d)
        # Remove sample dimension in output:
        logpdf_dd = lambda theta, y, d : approx_post.logpdf_del_d(theta[:,None,:], x=y, d=d)[:,0,:]
        # Remove sample dimension in output:
        logpdf_dy = lambda theta, y, d : approx_post.logpdf_del_x(theta[:,None,:], x=y, d=d)[:,0,:]
        return cls(logpdf=logpdf, logpdf_dd=logpdf_dd, logpdf_dy=logpdf_dy)

    @classmethod
    def from_jax_function(cls, jax_func, use_vmap=True, use_fwd=True):
        if use_fwd:
            grad = jax.jacfwd
        else:
            grad = jax.jacrev
        func_dict = {'logpdf': jax_func,
                     'logpdf_dy': grad(jax_func, argnums=1),
                     'logpdf_dd': grad(jax_func, argnums=2)}
        if use_vmap:
            for key, func in func_dict.items():
                func_dict[key] = jax.vmap(func, in_axes=(0,0,0))
        return cls(**func_dict)

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

        def logpdf_and_grads(theta, y, d, return_logpdf, return_dd, return_dy):
            # Assume y = g(theta, d) + noise
            outputs = {}
            if return_logpdf or return_dd:
                t_map = theta_map(theta, y, d)
                g_map = model.predict(t_map, d)
                g_dt_map = model.predict_dt(t_map, d)
                b = linearisation_constant(g_map, g_dt_map, t_map)
                mean, cov, icov = mean_cov_and_icov(y, t_map, g_dt_map, b)
            if return_logpdf:
                t_dim = cov.shape[0]
                outputs['logpdf'] = utils.gaussian_logpdf(theta, mean, cov, icov) 
            if return_dd or return_dy:
                g_dt_dt_map = model.predict_dt_dt(t_map, d)
            if return_dd:
                g_dd_map = model.predict_dd(t_map, d)
                g_dt_dd_map = model.predict_dt_dd(t_map, d)
                t_map_dd = theta_map_dd(y, g_map, g_dt_map, g_dd_map, g_dt_dt_map, g_dt_dd_map)
                mean_dd, cov_dd, icov_dd = \
                mean_cov_and_icov_dd(y, g_dt_map, g_dd_map, g_dt_dt_map, g_dt_dd_map, t_map, t_map_dd, cov, b)
                outputs['logpdf_dd'] = -0.5*(np.einsum("aijk,aji->ak", cov_dd, icov) + \
                                             np.einsum("aijk,ai,aj->ak", icov_dd, theta-mean, theta-mean) - \
                                             2*np.einsum("alk,ali,ai->ak", mean_dd, icov, theta-mean))
            if return_dy:
                t_map_dy = theta_map_dy(y, g_map, g_dt_map, g_dt_dt_map)
                mean_dy, cov_dy, icov_dy = mean_cov_and_icov_dy(y, g_dt_map, g_dt_dt_map, t_map, t_map_dy, cov, b)
                outputs['logpdf_dy'] = -0.5*(np.einsum("aijk,aji->ak", cov_dy, icov) + \
                                             np.einsum("ai,aijk,aj->ak", theta-mean, icov_dy, theta-mean) - \
                                             2*np.einsum("aik,aij,aj->ak", mean_dy, icov, theta-mean))
            return outputs        

        #
        #   Helper Functions
        #

        def theta_map(theta_0, y, d):
            return minimizer(map_loss_and_grad, theta_0, args=(y, d))

        def map_loss_and_grad(theta, y, d):
            y_pred = model.predict(theta, d)
            y_del_theta = model.predict_dt(theta, d)
            loss = np.einsum("ai,ij,aj->a", y-y_pred, noise_icov, y-y_pred) + \
                   np.einsum("ai,ij,aj->a", theta-prior_mean, prior_icov, theta-prior_mean)
            loss_del_theta = -2*np.einsum("aik,ij,aj->ak", y_del_theta, noise_icov, y-y_pred) + \
                              2*np.einsum("ij,aj->ai", prior_icov, theta-prior_mean)
            return loss, loss_del_theta
        
        def map_loss_dt_dt(y, g_map, g_dt_map, g_dt_dt_map):
            return 2*(prior_icov + np.einsum("ali,lk,akj->aij", g_dt_map, noise_icov, g_dt_map) \
                     - np.einsum("alij,lk,ak->aij", g_dt_dt_map, noise_icov, y-g_map))   

        def theta_map_dd(y, g_map, g_dt_map, g_dd_map, g_dt_dt_map, g_dt_dd_map):
            loss_dt_dt = map_loss_dt_dt(y, g_map, g_dt_map, g_dt_dt_map)
            loss_dt_dd = 2*(np.einsum("ali,lk,akj->aij", g_dt_map, noise_icov, g_dd_map) - \
                            np.einsum("alij,lk,ak->aij", g_dt_dd_map, noise_icov, y-g_map))
            return -1*np.linalg.solve(loss_dt_dt, loss_dt_dd)

        def linearisation_constant(g_map, g_dt_map, theta_map):
            return g_map - np.einsum("aij,aj->ai", g_dt_map, theta_map)

        def mean_cov_and_icov(y, theta_map, G, b):
            # G = partial_0 g(theta=theta_map(y,d), d) = g_dt_map
            inv_cov = np.einsum("aki,kl,alj->aij", G, noise_icov, G) + prior_icov
            cov = np.linalg.inv(inv_cov)
            mean_times_inv_cov = np.einsum("aj,jk,aki->ai", y-b, noise_icov, G) + np.einsum('i,ij->j', prior_mean, prior_icov)
            mean = np.einsum("ak,aki->ai", mean_times_inv_cov, cov)
            return mean, cov, inv_cov

        def mean_cov_and_icov_dd(y, G, g_dd_map, g_dt_dt_map, g_dt_dd_map, t_map, t_map_dd, cov, b):
            # G = partial_0 g(theta=theta_map(y,d), d) = g_dt_map
            # G_dd = partial_d (partial_0 g(theta=theta_map(y,d), d)):
            G_dd = g_dt_dd_map + np.einsum('aij,akli->aklj', t_map_dd, g_dt_dt_map)
            icov_dd = np.einsum("alik,lm,amj->aijk", G_dd, noise_icov, G) + \
                      np.einsum("ali,lm,amjk->aijk", G, noise_icov, G_dd)
            cov_dd = -1*np.einsum("ail,almk,amj->aijk", cov, icov_dd, cov)
            b_dd = np.einsum("akj,aik->aij", t_map_dd, G) + g_dd_map - \
                   np.einsum("aikj,ak->aij", G_dd, t_map) - \
                   np.einsum("aik,akj->aij", G, t_map_dd)
            mean_dd = np.einsum("akij,al,lm,amk->aij", cov_dd, y-b, noise_icov, G) -\
                      np.einsum("aki,alj,lm,amk->aij", cov, b_dd, noise_icov, G) +\
                      np.einsum("aki,al,lm,amkj->aij", cov, y-b, noise_icov, G_dd) +\
                      np.einsum("l,lk,akij->aij", prior_mean, prior_icov, cov_dd)
            return mean_dd, cov_dd, icov_dd

        def theta_map_dy(y, g_map, g_dt_map, g_dt_dt_map):
            loss_dt_dt = map_loss_dt_dt(y, g_map, g_dt_map, g_dt_dt_map)    
            loss_dt_dy = -2*np.einsum('ik,akj->aij', noise_icov, g_dt_map)
            return -1*np.linalg.solve(loss_dt_dt, loss_dt_dy)
        
        def mean_cov_and_icov_dy(y, G, g_dt_dt_map, t_map, t_map_dy, cov, b):
            # G = partial_0 g(theta=theta_map(y,d), d) = g_dt_map
            # G_dy = partial_y (partial_0 g(theta=theta_map(y,d), d)):
            G_dy = np.einsum('aij,akli->aklj', t_map_dy, g_dt_dt_map)
            icov_dy = np.einsum("alik,lm,amj->aijk", G_dy, noise_icov, G) + \
                      np.einsum("ali,lm,amjk->aijk", G, noise_icov, G_dy)
            cov_dy = -1*np.einsum("ail,almk,amj->aijk", cov, icov_dy, cov)
            b_dy = np.einsum("akj,aik->aij", t_map_dy, G) - \
                   np.einsum("aikj,ak->aij", G_dy, t_map) - \
                   np.einsum("aik,akj->aij", G, t_map_dy)
            y_minus_b_dy = np.identity(b_dy.shape[-1]) - b_dy
            mean_dy = np.einsum("akij,al,lm,amk->aij", cov_dy, y-b, noise_icov, G) + \
                      np.einsum("aki,alj,lm,amk->aij", cov, y_minus_b_dy, noise_icov, G) + \
                      np.einsum("aki,al,lm,amkj->aij", cov, y-b, noise_icov, G_dy) + \
                      np.einsum("l,lk,akij->aij", prior_mean, prior_icov, cov_dy)
            return mean_dy, cov_dy, icov_dy

        return cls(logpdf_and_grads=logpdf_and_grads)

class Joint(Distribution):

    def __init__(self, sample=None):
        self._func_dict = {'sample': sample}

    @classmethod
    def from_prior_and_likelihood(cls, prior, likelihood):
        def sample(d, num_samples, rng):
            theta = prior.sample(num_samples, rng)
            y = likelihood.sample(theta, d, num_samples, rng)
            return theta, y
        return cls(sample=sample)

    def sample(self, d, num_samples, rng=None):
        if 'sample' not in self._func_dict:
            return AttributeError('Sampling function not specified.')
        d = utils._preprocess_inputs(d=d)
        theta, y = self._func_dict['sample'](d, num_samples, rng)
        return {'theta': theta.reshape(num_samples, theta.shape[-1]), 
                'y': y.reshape(num_samples, y.shape[-1])}