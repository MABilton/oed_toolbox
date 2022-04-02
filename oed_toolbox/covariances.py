import numpy as np

def fisher_information_samples(loglike_del_2_theta, loglike_del_d, loglike_del_2_theta_del_d, sample_likelihood, theta_dim, d_dim):
    
    def fisher_information_and_grad(d, theta_estimate, num_samples, rng):
        
        y = sample_likelihood(theta_estimate, d, num_samples, rng)

        ll_del_d = loglike_del_d(y, theta_estimate, d).reshape(num_samples, d_dim)
        ll_del_2_theta = loglike_del_2_theta(y, theta_estimate, d).reshape(num_samples, theta_dim, theta_dim)
        ll_del_2_theta_del_d = loglike_del_2_theta_del_d(y, theta_estimate, d).reshape(num_samples, theta_dim, theta_dim, d_dim)

        fisher_info = ll_del_2_theta
        fisher_info_del_d = np.einsum('ak,aij->aijk', ll_del_d, ll_del_2_theta) + ll_del_2_theta_del_d

        return {'fisher_information': fisher_info, 'fisher_information_del_d': fisher_info_del_d,
                'control variates': {'ll_del_2_theta': ll_del_2_theta, 'll_del_2_theta_del_d': ll_del_2_theta_del_d}}
    
    return fisher_information_and_grad

def predictive_covariance(prediction_del_theta, prediction_del_theta_del_d):
    
    def predictive_covariance_and_grad(d, theta_estimate, fisher_info, fisher_info_del_d, pred_dim, theta_dim, d_dim):
        
        inv_fisher_info = np.linalg.inv(fisher_info).reshape(theta_dim, theta_dim)
        # Derivative of inverse fisher info matrix - see Eqn (59) in Matrix cookbook (https://www2.imm.dtu.dk/pubdb/edoc/imm3274.pdf):
        inv_fisher_info_del_d = \
        -1*np.einsum('ij,jkm,kl->ilm', inv_fisher_info, fisher_info_del_d.reshape(theta_dim, theta_dim, d_dim), inv_fisher_info)

        pred_del_theta = prediction_del_theta(theta_estimate, d).reshape(pred_dim, theta_dim)
        pred_del_theta_del_d = prediction_del_theta_del_d(theta_estimate, d).reshape(pred_dim, theta_dim, d_dim)

        pred_cov = np.einsum('ij,jk,kl->il', pred_del_theta, inv_fisher_info, pred_del_theta.T)
        pred_cov_del_d = np.einsum('ijm,jk,kl->ilm', pred_del_theta_del_d, inv_fisher_info, pred_del_theta.T) +\
                         np.einsum('ij,jkm,kl->ilm', pred_del_theta, inv_fisher_info_del_d, pred_del_theta.T) +\
                         np.einsum('ij,jk,klm->ilm', pred_del_theta, inv_fisher_info, pred_del_theta_del_d.T)

        return pred_cov, pred_cov_del_d

    return predictive_covariance_and_grad