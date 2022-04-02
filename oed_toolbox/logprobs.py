

class LogLikelihood:

    @classmethod
    def from_model_plus_constant_gaussian_noise(cls, y_dim, theta_dim, d_dim, model, model_dt, model_dd, model_dt_dt, model_dt_dd, noise_cov):

        def loglikelihood(y, theta, d):
            return 

        def loglikelihood_dt(y, theta, d):
            pass

        def loglikelihood_dt_dt(y, theta, d):
            pass
        
        def loglikelihood_dd(y, theta, d):
            pass

        def loglikelihood_dt_dd(y, theta, d):
            pass

        def loglikelihood_dt_dt_dd(y, theta, d):
            pass

        def sample(theta, d, num_samples, rng):
            pass

        return cls()




    def __init__(y_dim, theta_dim, d_dim, num_obs, sample=None, logprob=None, logprob_del_theta=None, logprob):
        self._func_dict = func_dict
        self._ydim = y_dim
        self._theta_dim = theta_dim
        self._d_dim = d_dim
        self._num_obs = num_obs

    def __get__(self, key):
        return self._func_dict[key]

    @property
    def functions():
        return self._func_dict


class Prior:
    
    @classmethod
    def gaussian():

        def sample():
            pass

        def logprob():
            pass

        def logprob_del_theta():
            pass
        
        return cls()

    def __init__():
        pass

    def __get__(self, key):
        return self._func_dict[key]

class Joint:
    
    @classmethod
    def from_prior_and_likelihood(prior, likelihood):
        def sample(num_samples, rng):
            theta_samples = 
            y_samples = 
            return theta_samples, y_samples

        logprob = lambda 
        logprob_dt = lambda

        return cls()

class Posterior:
    pass