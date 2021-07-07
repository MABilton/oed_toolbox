from math import inf, pi
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
import numpy as np
from numpy.random import multivariate_normal as mvn
from scipy.optimize import minimize

def create_local_linear_funcs(model, noise_cov, prior_mean, prior_cov):
    # Create prior function to sample from:
    sample_prior = create_sample_prior(prior_mean, prior_cov)
    # Create likelihood function to sample from:
    sample_likelihood = create_sample_likelihood(model, noise_cov)
    # Create function to compute log_post and gradients of log_post and log_like:
    log_probs_and_grads = create_log_probs_and_grads()
    return (sample_prior, sample_likelihood, log_probs_and_grads)

# Function which creates function to sample from prior:
def create_sample_prior(prior_mean, prior_cov):
    def sample_prior(num_samples):
        return mvn(prior_mean, prior_cov, size=num_samples)
    return sample_prior

# Function which creates function to sample from likelihood:
def create_sample_likelihood(model, noise_cov):
    # Can't vectorise sampling with vmap - need to use loop:
    def sample_likelihood(d, theta_samples):
        samples = jnp.empty()
        for i in range(theta_samples.shape[0]):
            mean = model(d, theta_samples[i,:])
            sample_i = mvn(mean, noise_cov, size=1)
            samples = jnp.vstack([samples, sample_i]) if samples.size else jnp.array(sample_i)
        return samples
    return sample_likelihood

# Function which creates function to compute MAP of posterior:
def create_compute_map(model, model_grad_theta, noise_cov, prior_mean, prior_cov, theta_bounds):
    # Loss function to minimise to find MAP:
    def map_loss_and_grad(theta, y, d):
        # Compute loss:
        y_mean = model(d, theta)
        noise_cov_inv_times_y_diff = jlinalg.solve(noise_cov, y-y_mean)
        prior_cov_inv_times_theta_diff = jlinalg.solve(prior_cov, theta-prior_mean)
        loss = (y-y_mean).T @ noise_cov_inv_times_y_diff + (theta-prior_mean).T @ prior_cov_inv_times_theta_diff
        # Compute gradient of loss wrt theta:
        del_ymean_del_theta = model_grad_theta(d, theta)
        loss_grad = -2*del_ymean_del_theta*noise_cov_inv_times_y_diff + 2*prior_cov_inv_times_theta_diff
        return (loss, loss_grad) 
    # Function to compute MAP and MAP Jacobian for a SINGLE sample:
    def compute_map_single_sample(d, theta, y, num_repeats=3):
        best_loss = inf
        for i in range(num_repeats):
            theta_0 = np.random.rand(theta_bounds.shape[0])*(theta_bounds[:,1] - theta_bounds[:,0]) + theta_bounds[:,0]
            opt_result = minimize(map_loss_and_grad, theta_0, bounds=theta_bounds, method="L-BFGS-B", jac=True)
            if opt_result["fun"] < best_loss:
                best_loss, best_theta = opt_result["fun"], opt_result["x"]
        return best_theta
    # Function which finds MAP and Jacobian of MAP for ALL samples:
    def compute_map(d, prior_samples, like_samples):
        theta_map = jnp.empty()
        # Need to manually iterate over samples here
        for i in range(prior_samples.shape[0]):
            theta_i = compute_map_single_sample(d, prior_samples[i,:], like_samples[i,:])
            theta_map = jnp.vstack((theta_map, theta_i)) if theta_map.size else jnp.array(theta_i)
        return theta_map
    return compute_map

# Function which creates function to compute the gradient of the MAP wrt theta:
# TODO: Double check dimensions + assert statements
def create_compute_map_grad(model, model_del_theta, model_del_d, model_del_2_theta, model_del_theta_d, noise_cov, prior_cov):
    # Function which computes gradient of MAP for a SINGLE sample:
    def compute_map_grad_single_sample(d, theta_map, y):
        # Compute common terms:
        del_y = y-model(d, theta_map)
        inv_cov_del_y = jlinalg.solve(noise_cov, del_y)
        model_del_theta_val = model_del_theta(d, theta_map)
        # Compute 2nd derivative of MAP loss wrt theta:
        loss_del_2_theta = -2*model_del_2_theta(d,theta_map) @ inv_cov_del_y \
             + 2*model_del_theta_val @ jlinalg.solve(noise_cov, model_del_theta_val.T) + 2*jlinalg.inv(prior_cov)
        # Compute mixed derivative of MAP loss wrt theta AND d:
        loss_del_theta_d = -2*model_del_theta_d(d,theta_map) @ inv_cov_del_y \
            + 2*model_del_theta_val @ jlinalg.solve(noise_cov, model_del_d(d, theta_map).T)
        # Return gradient of MAP point:
        return -1*jlinalg.solve(loss_del_2_theta, loss_del_theta_d)
    # Function which computes gradient of MAP for ALL samples:
    def compute_map_grad(map_values, d, like_samples):
        # Compute gradient of MAP for each values:
        map_grads = jnp.empty()
        for i in range(map_values.shape[0]):
            # Compute gradient of MAP for single value:
            map_grad_i = compute_map_grad_single_sample(d, map_values[i,:], like_samples[i,:])
            # Store this gradient:
            map_grads = jnp.stack((map_grads, map_grad_i),axis=2) if map_grads.size else jnp.atleast_2d(map_grads)
        return map_grads
    return compute_map_grad

def create_linearise_model(model, model_del_theta):
    def linearise_model_single_sample(theta_map, d):
        G = model_del_theta(d, theta_map)
        b = model(d, theta_map) - G @ theta_map
        return (G, b)
    def linearise_model(map_values, d):
        G_store, b_store = jnp.empty(), jnp.empty()
        for i in range(map_values.shape[0]):
            G_i, b_i = linearise_model_single_sample(map_values[i,:], d)
            G_store = jnp.stack((G_store, G_i),axis=2) if G_store.size else jnp.atleast_2d(G_i)
            b_store = jnp.stack((b_store, b_i),axis=2) if b_store.size else jnp.atleast_2d(b_i)
        return (G_store, b_store)
    return linearise_model

def create_compute_log_post(noise_cov, prior_cov, prior_mean):
    def compute_log_post_single_sample(theta, y, G, b, k):
        post_cov = G.T @ jlinalg.solve(noise_cov, G) + jlinalg.inv(prior_cov)
        post_mean = post_cov @ G.T @ jlinalg.solve(noise_cov, y - b) + post_cov @ jlinalg.solve(prior_cov, prior_mean)
        return -k/2*jnp.log(2*pi) + -1/2*jnp.log(jlinalg.det(post_cov)) \
            - 1/2*(theta-post_mean).T @ jlinalg.solve(post_cov, (theta-post_mean))
    def compute_log_post(prior_samples, like_samples, G_store, b_store):
        k = prior_mean.size
        log_post_store = jnp.empty()
        for i in range(prior_samples.shape[0]):
            log_post_i = compute_log_post_single_sample(prior_samples[i,:], like_samples[i,:], G_store[:,:,i], b_store[:,:,i], k)
            log_post_store = jnp.stack((log_post_store, log_post_i), axis=1) if log_post_store.size else jnp.atleast_1d(log_post_i)
        return log_post_store
    return compute_log_post

def create_compute_log_prob_grads():
    def compute_log_prob_grads_single_sample(d, theta, y, G, b, k):
        # Compute common terms:
        model_val = model(d, theta)
        model_del_d_val = model_del_d(d, theta)
        # Compute gradient of log_like:
        log_like_grad = model_del_d_val @ jlinalg.solve(noise_cov, y-model_val)
        # Compute gradient of log_post:



def create_log_probs_and_grads():
    # Create functions required to create the log_probs_and_grad function:
    compute_map = create_compute_map(model, model_grad_theta, noise_cov, prior_mean, prior_cov, theta_bounds)
    compute_map_grad = create_compute_map_grad(model, model_del_theta, model_del_d, model_del_2_theta, model_del_theta_d, noise_cov, prior_cov)
    linearise_model = create_linearise_model(model, model_del_theta)
    compute_log_post = create_compute_log_post(noise_cov, prior_cov, prior_mean)
    def log_probs_and_grads(d, prior_samples, like_samples):
        # Compute posterior MAP for these samples:
        theta_map = compute_map(d, prior_samples, like_samples)
        # Compute linerisations of model about these MAP points:
        G, b = linearise_model(theta_map, d)
        # Compute log posterior using linearisations:
        log_post = compute_log_post(prior_samples, like_samples, G, b)
        # Compute gradients of MAP points wrt d:
        map_grad = compute_map_grad(d, prior_samples, like_samples, theta_map)
        # Compute log prob gradients wrt d using linearisations and map_grads:
        log_like_grad, log_post_grad = compute_log_prob_grads()
        return (log_post, log_like_grad, log_post_grad)
    return log_probs_and_grads