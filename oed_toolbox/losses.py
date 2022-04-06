import numpy as np
from . import uncertainties, utils

#
#   Approximate Posterior Entropy
#

class APE:

    def __init__(self, joint, likelihood, posterior, apply_control_variates=True):
        self._loss_and_grad = self._create_loss(joint, likelihood, posterior, apply_control_variates)

    def __call__(self, d, num_samples, rng=None, return_grad=True):
        return self._loss_and_grad(d, num_samples, rng, return_grad)

    @classmethod
    def from_model_plus_constant_gaussian_noise(cls, model, minimizer, prior_mean, prior_cov, noise_cov, apply_control_variates=True):
        prior = uncertainties.Prior.gaussian(prior_mean, prior_cov)
        likelihood = uncertainties.Likelihood.from_model_plus_constant_gaussian_noise(model, noise_cov)
        joint = uncertainties.Joint.from_prior_and_likelihood(prior, likelihood)
        approx_posterior = uncertainties.Posterior.laplace_approximation(model, minimizer, noise_cov, prior_mean, prior_cov)
        return cls(joint, likelihood, approx_posterior, apply_control_variates)

    @staticmethod
    def _create_loss(joint, likelihood, posterior, apply_control_variates):
        
        def ape_and_grad(d, num_samples, rng, return_grad):

            theta, y = joint.sample(d, num_samples, rng)
            # Remove batch dimension:
            theta, y = theta[:,0,:], y[:,0,:]
            post_vals = posterior.logpdf(theta, y, d, return_dd=return_grad)
            loss_samples = post_vals['logpdf']

            # Need to compute like_grad if we're applying control variates:
            if return_grad or apply_control_variates:
                like_grad = likelihood.logpdf(y, theta, d, return_logpdf=False, return_dd=True)['logpdf_dd']
            if return_grad:
                grad_samples = np.einsum('a,ai->ai', post_vals['logpdf'], like_grad) + post_vals['logpdf_dd']

            if apply_control_variates:
                avging_func = lambda val : utils.apply_control_variates(val, cv=like_grad)
            else:
                avging_func = lambda val : np.mean(val, axis=0)

            loss = -1*avging_func(loss_samples)
            if return_grad:
                loss_del_d = -1*avging_func(grad_samples)

            return loss if not return_grad else (loss, loss_del_d)

        return ape_and_grad

#
#   'Alphabet' Optimal Criteria
#

class _Alphabet:
    
    def __init__(self, cov_func):
        self._loss_and_grad = self._create_loss_and_grad(cov_func)

    def __call__(self, d, theta_estimate, num_samples, rng=None, return_grad=True):
        return self._loss_and_grad(d, theta_estimate, num_samples, rng, return_grad)

class D_Optimal(_Alphabet):

    @staticmethod
    def _create_loss_and_grad(cov_func):
        def loss_and_grad(d, theta_estimate, num_samples, rng, return_grad):
            cov_vals = cov_func(d, theta_estimate, num_samples, rng, return_dd=return_grad)
            cov = cov_vals['cov']
            loss = -1*np.linalg.det(cov)
            if return_grad:
                # Derivative of det(M) wrt M - see Eqn (49) in Matrix Cookbook (https://www2.imm.dtu.dk/pubdb/edoc/imm3274.pdf):
                cov_dd = cov_vals['cov_dd']
                loss_del_cov = loss*np.linalg.inv(cov).T
                loss_del_d = jnp.einsum('ij,ijk->k', loss_del_cov, cov_dd)
            return loss if not return_grad else loss, loss_del_d
        return loss_and_grad

class A_Optimal(_Alphabet):

    @staticmethod
    def _create_loss_and_grad(cov_func):
        def loss_and_grad(d, theta_estimate, num_samples, rng, return_grad):
            cov_vals = cov_func(d, theta_estimate, num_samples, rng, return_dd=return_grad)
            cov = cov_vals['cov']
            inv_cov = np.linalg.inv(cov)
            loss = np.linalg.trace(cov)
            if return_grad:
                # Derivative of tr(M^-1) wrt M = -((M^-1)^T)@((M^-1)^T) - substitute A = B = I into Eqn (124) in Matrix Cookbook:
                cov_dd = cov_vals['cov_dd']
                loss_del_cov = -1*np.einsum('ji,kj->ik', inv_cov, inv_cov)
                loss_del_d = np.einsum('ij,ijk->k', loss_del_cov, cov_dd)
            return loss if not return_grad else loss, loss_del_d
        return loss_and_grad

class E_Optimal(_Alphabet):

    @staticmethod
    def _create_loss_and_grad(cov_func):
        def loss_and_grad(d, theta_estimate, num_samples, rng, return_grad):
            cov_vals = cov_func(d, theta_estimate, num_samples, rng, return_dd=return_grad)
            cov = cov_vals['cov']
            eigvals, eigvecs = np.linalg.eigh(cov)
            loss, min_eigvec = eigvals[0], eigvecs[:,0]
            if return_grad:
                # Derivative of eigenvalue wrt matrix - see https://math.stackexchange.com/questions/2588473/derivatives-of-eigenvalues
                cov_dd = cov_vals['cov_dd']
                loss_del_cov = np.einsum('i,j->ij', min_eigvec, min_eigvec)
                loss_del_d = np.einsum('ij,ijk->k', loss_del_cov, cov_dd)
            return loss if not return_grad else loss, loss_del_d
        return loss_and_grad
