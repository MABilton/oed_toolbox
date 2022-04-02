from math import pi
import numpy as np
from scipy.stats import multivariate_normal as mvn

#
#   Exact APE Loss
#

class APE_Loss:

    @classmethod
    def from_logprobs(cls, posterior, likelihood, joint, expectation, d_dim):

        def ape_and_grad(d, num_samples, state=None):
            theta, y = sample_joint(d, num_samples, state)
            post = posterior(theta, y, d).reshape(-1)
            like_grad = likelihood_del_d(theta, y, d).reshape(-1, d_dim)
            post_grad = posterior_del_d(theta, y, d).reshape(-1, d_dim)
            loss_samples = post
            grad_samples = np.einsum('a,ai->ai', post, like_grad) + post_grad
            return loss_samples, grad_samples

        return cls(ape_and_grad, expectation)

    @classmethod
    def from_linearised_model_plus_constant_gaussian_noise(prior,):


    def __init__():
        pass


    def __eval__(d, num_samples, rng=None):
        return self._loss_and_grad(d, num_samples, rng)





#
#   Linearisation Approximation of APE
#

def by_linearising_model_with_gaussian_noise(d_dim, y_dim, theta_dim, model, model_del_theta, model_del_d, model_del_2_theta, model_del_theta_del_d, noise_cov, prior_mean, prior_cov, minimizer):

    prior_icov = np.linalg.inv(prior_cov)
    noise_icov = np.linalg.inv(noise_cov)

    #
    #   Main function to evaluate loss and gradient:
    #

    def loss_and_grad_samples(d, num_samples, state=None):
        
        # Sample theta and y from joint:
        theta = sample_prior(num_samples, state).reshape(-1, theta_dim)
        y_pred = model(theta, d).reshape(-1, y_dim)
        y = sample_likelihood(y_pred, num_samples, state).reshape(-1, y_dim)

        # Compute MAP for each sample:
        theta_0 = theta
        theta_map = minimizer(map_loss_and_grad, theta_0, y, d).reshape(-1, theta_dim)
        
        # Compute model values and derivatives at theta_map point:
        y_map = model(theta_map, d).reshape(-1, y_dim)
        y_map_del_theta = model_del_theta(theta_map, d).reshape(-1, y_dim, theta_dim)
        y_map_del_d = model_del_d(theta_map, d).reshape(-1, y_dim, d_dim)
        y_map_del_2_theta = model_del_2_theta(theta_map, d).reshape(-1, y_dim, theta_dim, theta_dim)
        y_map_del_theta_del_d = model_del_theta_del_d(theta_map, d).reshape(-1, y_dim, theta_dim, d_dim)
        
        # Compute gradient of MAP wrt d:
        theta_map_del_d = \
        theta_map_grad(y_map, y_theta_map, y_d_map, y_theta_theta_map, y_theta_d_map)

        # b is the constant term in our affine approximation of the model, i.e.:
        # y = model(theta, d) ≈ model_del_theta(theta_map, d) * theta + b
        b = y_map - np.einsum("aij,aj->ai", y_map_del_theta, theta_map)

        # Compute mean and inverse covariance of (approximate) normal posterior:
        post_mean, post_cov, post_icov = posterior_mean_and_cov(y, theta_map, y_map, y_map_del_theta, b)

        # Compute log posterior:
        post = approx_posterior(theta, post_mean, post_cov, post_icov)
        
        # Compute gradient of posterior mean wrt d and posterior cov wrt d:
        post_mean_del_d, post_cov_del_d, post_icov_del_d = \
        posterior_mean_and_cov_del_d(y, theta_map, theta_map_del_d, post_cov, b, y_map_del_theta, y_map_del_d, y_map_del_theta_del_d)
        
        # Compute gradient of log posterior wrt d:
        post_del_d = approx_posterior_del_d(theta, post_mean, post_icov, post_mean_del_d, post_cov_del_d, post_icov_del_d)

        # Compute gradient of log likelihood wrt d: 
        like_del_d = likelihood_del_d(theta, y, y_pred, d).reshape(-1, d_dim)

        loss_samples = post
        grad_samples = np.einsum('a,ai->ai', post, like_del_d) + post_del_d

        return loss_samples, grad_samples

    #
    #   Helper methods called by loss_and_grad_samples:
    #

    def sample_prior(num_samples, state):
        return mvn.rvs(mean=prior_mean, cov=prior_cov, size=num_samples, random_state=state)

    def sample_likelihood(y_obs, num_samples, state):
        return mvn.rvs(mean=y_obs, cov=noise_cov, size=num_samples, random_state=state)

    def map_loss_and_grad(theta, y, d):
        y_pred = model(theta, d).reshape(-1, y_dim)
        y_del_theta = model_del_theta(theta, d).reshape(-1, y_dim, theta_dim)
        z_theta = np.einsum('ij,aj->ai', prior_icov, theta-prior_mean)
        z_y = np.einsum('ij,aj->ai', noise_icov, y-y_pred)
        loss = np.einsum("ai,aij->aj", theta, z_theta) + np.einsum('ai,aij->aj', y-y_pred, z_y) 
        loss_del_theta = 2*z_theta - 2*np.einsum("aik,aij->ak", y_del_theta, z_y)
        return (loss, loss_del_theta)

    def theta_map_grad(y, y_map, y_map_del_theta, y_map_del_d, y_map_del_2_theta,y_map_del_theta_del_d):

        map_loss_del_2_theta = \
            2*(inv_prior - \
            np.einsum("alij,lk,ak->aij", y_map_del_2_theta, noise_icov, y-y_map) - \
            np.einsum("ali,lk,akj->aij", y_map_del_theta, noise_icov, y_map_del_theta))

        map_loss_del_theta_del_d = \
        2*(np.einsum("ali,lk,akj->aij", y_map_del_theta, inv_noise, y_map_del_d) - \
           np.einsum("alij,lk,ak->aij", y_map_del_theta_del_d, inv_noise, y-y_map))
            
        theta_map_del_d = -1*np.linalg.solve(map_loss_del_2_theta, map_loss_del_theta_del_d)

        return theta_map_del_d

    def posterior_mean_and_cov(y, theta_map, y_map, y_map_del_theta, b):
        inv_cov = np.einsum("aki,kl,alj->aij", y_map_del_theta, inv_noise, y_map_del_theta) + inv_prior
        cov = linalg.inv(inv_cov)
        mean_times_inv_cov = \
            np.einsum("aj,jk,aki->ai", y-b, inv_noise, y_map_del_theta) + np.einsum('i,ij->j', prior_mean, inv_prior)
        mean = np.einsum("ak,aki->ai", mean_times_inv_cov, cov)
        return mean, cov, inv_cov
    
    def approx_posterior(theta, post_mean, post_cov, post_icov):
        return -0.5*(theta_dim*np.log(2*pi) + np.log(np.linalg.det(post_cov)) + \
                     np.einsum("ai,aij,aj->a", theta-post_mean, post_icov, theta-post_mean))

    def posterior_mean_and_cov_del_d(y, theta_map, theta_map_del_d, post_cov, b, y_map_del_theta, y_map_del_d, y_map_del_theta_del_d):
        
        post_icov_del_d = \
            np.einsum("alik,lm,amj->aijk", y_map_del_theta_del_d, noise_icov, y_map_del_theta) + \
            np.einsum("ali,lm,amjk->aijk", y_map_del_theta, noise_icov, y_map_del_theta_del_d)
        
        post_cov_del_d = -1*np.einsum("ail,almk,amj->aijk", post_cov, post_icov_del_d, post_cov)
        
        b_del_d = np.einsum("akj,aik->aij", theta_map_del_d, y_map_del_theta) - \
                  np.einsum("aikj,ak->aij", y_map_del_theta_del_d, theta_map) - \
                  np.einsum("aik,akj->aij", y_map_del_theta, theta_map_del_d) + \
                  y_map_del_d

        post_mean_del_d = np.einsum("akij,al,lm,amk->aij", post_cov_del_d, y-b, noise_icov, y_map_del_theta) -\
                          np.einsum("aki,alj,lm,amk->aij", post_cov, b_del_d, noise_icov, y_map_del_theta) +\
                          np.einsum("aki,al,lm,amkj->aij", post_cov, y-b, noise_icov, y_map_del_theta_del_d) +\
                          np.einsum("l,lk,akij->aij", prior_mean, prior_cov, post_cov_del_d)

        return post_mean_del_d, post_cov_del_d, post_icov_del_d

    def approx_posterior_del_d(theta, post_mean, post_icov, post_mean_del_d, post_cov_del_d, post_icov_del_d):
        return -0.5*(np.einsum("aijk,aji->ak", post_cov_del_d, post_icov) + \
                     np.einsum("aijk,ai,aj->ak", post_icov_del_d, theta-post_mean, theta-post_mean) - \
                     2*np.einsum("alk,ali,ai->ak", post_mean_del_d, post_icov, theta-post_mean))

    def likelihood_del_d(theta, y, y_pred, d):
        y_del_d = model_del_d(theta, d).reshape(-1, y_dim, d_dim)
        return np.einsum("ajk,jl,al->ak", y_del_d, noise_icov, y-y_pred)

    return loss_and_grad_samples