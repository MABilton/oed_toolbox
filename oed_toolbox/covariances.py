import numpy as np
from . import utils

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
                outputs = utils._attempt_func_call(cov, outputs, func_name='cov', args=(d, theta_estimate, num_samples, rng))
            if return_dd:
                outputs = utils._attempt_func_call(cov_dd, outputs, func_name='cov_dd', args=(d, theta_estimate, num_samples, rng))
            return outputs
        return _create_cov_and_grads

    def __call__(self, d, theta_estimate, num_samples=None, rng=None, return_cov=True, return_dd=True, samples=None):
        if (num_samples is None) and (samples is None):
            raise ValueError('Must specify either num_samples or samples.')
        return self._func_dict['cov_and_grads'](d, theta_estimate, num_samples, rng, return_dd, samples)

class FisherInformation(Covariance):

    def __init__(self, likelihood, apply_control_variates=True, use_reparameterisation=False):
        if use_reparameterisation:
            cov_and_grad =  self._create_reparameterisation_fisher_info(likelihood, apply_control_variates)
        else:
            cov_and_grad = self._create_fisher_info(likelihood, apply_control_variates)
        super().__init__(cov_and_grads=cov_and_grad)

    @staticmethod
    def _create_reparameterisation_fisher_info(likelihood, apply_control_variates):

        def cov_and_grad(d, theta, num_samples, rng, return_dd, samples):
            outputs = {}
            if samples is None:
                epsilon = likelihood.sample_base(num_samples, rng)
            else:
                epsilon = samples
            transform = likelihood.transform(epsilon, theta, d, return_dd)
            like_vals = likelihood.logpdf(transform['y'], theta, d, return_logpdf=False, return_dt=True, return_dt_dy=return_dd, return_dt_dd=return_dd)
            outputs['cov'] = np.einsum('ai,aj->aij', like_vals['logpdf_dt'], like_vals['logpdf_dt'])
            if return_dd:
                ll_dt_dd = np.einsum('ajk,aij->aik', transform['y_dd'], like_vals['logpdf_dt_dy']) + like_vals['logpdf_dt_dd']
                outputs['cov_dd'] = 2*np.einsum('aik,aj->aijk', ll_dt_dd, like_vals['logpdf_dt'])
            for key, val in outputs.items():
                outputs[key] = np.mean(val, axis=0)
            return outputs

        return cov_and_grad

    @staticmethod
    def _create_fisher_info(likelihood, apply_control_variates):

        def cov_and_grad(d, theta, num_samples, rng, return_dd, samples):
            compute_dd = return_dd or apply_control_variates
            outputs = {}
            if samples is None:
                y = likelihood.sample(theta, d, num_samples, rng)
            else:
                y = samples
            like_vals = likelihood.logpdf(y, theta, d, return_logpdf=False, return_dt=True, return_dt_dd=return_dd, return_dd=compute_dd)
            outputs['cov'] = np.einsum('ai,aj->aij', like_vals['logpdf_dt'], like_vals['logpdf_dt'])               
            if return_dd:
                outputs['cov_dd'] = \
                np.einsum('ak,ai,aj->aijk', like_vals['logpdf_dd'], like_vals['logpdf_dt'], like_vals['logpdf_dt']) + \
                np.einsum('aik,aj->aijk', like_vals['logpdf_dt_dd'], like_vals['logpdf_dt']) + \
                np.einsum('ai,ajk->aijk', like_vals['logpdf_dt'], like_vals['logpdf_dt_dd'])
            for key, val in outputs.items():
                if apply_control_variates:
                    outputs[key] = utils.apply_control_variates(val, cv=like_vals['logpdf_dd'])
                else:
                    outputs[key] = np.mean(val, axis=0)
            return outputs

        return cov_and_grad

class PredictiveCovariance(Covariance):

    def __init__(self, model, fisher_information):
        cov_and_grad = self._create_predictive_variance(model, fisher_information)
        super().__init__(cov_and_grads=cov_and_grad)
        
    @staticmethod
    def _create_predictive_variance(model, fisher_information):
        def cov_and_grad(d, theta_estimate, num_samples, rng, return_dd, samples):
            outputs = {}
            cov_and_grad = fisher_information(d, theta_estimate, num_samples, rng, return_cov=True, return_dd=return_dd, samples=samples)
            inv_fisher_info = np.linalg.inv(cov_and_grad['cov'])
            # Remove batch dimension:
            y_dt = model.predict_dt(theta_estimate, d)[0,:]
            outputs['cov'] = np.einsum('ij,jk,kl->il', y_dt, inv_fisher_info, y_dt.T)
            if return_dd:
                # Remove batch dimension:
                y_dt_dd = model.predict_dt_dd(theta_estimate, d)[0,:]
                # Derivative of inverse fisher info matrix - see Eqn (59) in Matrix cookbook (https://www2.imm.dtu.dk/pubdb/edoc/imm3274.pdf):
                fisher_info_dd = cov_and_grad['cov_dd']
                inv_fisher_info_dd = -1*np.einsum('ij,jkm,kl->ilm', inv_fisher_info, fisher_info_dd, inv_fisher_info)
                outputs['cov_dd'] = 2*np.einsum('ijm,jk,kl->ilm', y_dt_dd, inv_fisher_info, y_dt.T) + \
                                    np.einsum('ij,jkm,kl->ilm', y_dt, inv_fisher_info_dd, y_dt.T)
            return outputs
        return cov_and_grad