import numpy as np
from numpy.random import multivariate_normal as mvn
# Set seed for reproducability:
np.random.seed(42)

def create_sampling_funcs(model, noise_cov, prior_mean, prior_cov):
    # Create prior function to sample from:
    sample_prior = create_sample_prior(prior_mean, prior_cov)
    # Create likelihood function to sample from:
    sample_likelihood = create_sample_likelihood(model, noise_cov)
    return (sample_prior, sample_likelihood)

def create_sample_prior(prior_mean, prior_cov):
    def sample_prior(num_samples):
        return mvn(prior_mean, prior_cov, size=num_samples)
    return sample_prior

# See https://juanitorduz.github.io/multivariate_normal/ for theory:
def create_sample_likelihood(model, noise_cov):
    # Sample from zero mean, unit variance Gaussian, then transform for each sample:
    def sample_likelihood(d, theta_samples):
        # Compute means -> means.shape = (n_samples, y_dim):
        means = model(theta_samples, d)
        # Compute number of zero mean Gaussian samples to draw -> samples.shape = (n_samples, y_dim):
        y_dim, n_samples = means.shape[1], means.shape[0]
        samples = mvn(np.zeros(y_dim), noise_cov, size=n_samples)
        # Transform samples:
        return samples + means
    return sample_likelihood