
import numpy as np

def d_optimal():
    def loss_and_grad(cov, cov_del_d):
        loss = -1*np.linalg.det(cov)
        # Derivative of det(M) wrt M - see Eqn (49) in Matrix Cookbook (https://www2.imm.dtu.dk/pubdb/edoc/imm3274.pdf):
        loss_del_cov = loss*np.linalg.inv(cov).T
        loss_del_d = jnp.einsum('ij,ijk->k', loss_del_cov, cov_del_d)
        return loss, loss_del_d
    return loss_and_grad

def a_optimal():
    def loss_and_grad(cov, cov_del_d):
        loss = -1*np.linalg.det(cov)
        inv_cov = np.linalg.inv(cov)
        loss = np.linalg.trace(cov)
        # Derivative of tr(M^-1) wrt M = -((M^-1)^T)@((M^-1)^T) - substitute A = B = Identity into Eqn (124) in Matrix Cookbook:
        loss_del_cov = -1*np.einsum('ji,kj->ik', inv_cov, inv_cov)
        loss_del_d = np.einsum('ij,ijk->k', loss_del_cov, cov_del_d)
        return (loss, loss_del_d)
    return loss_and_grad

def e_optimal(cov, cov_del_d):
    def loss_and_grad():
        eigvals, eigvecs = np.linalg.eigh(cov)
        loss, min_eigvec = eigvals[0], eigvecs[:,0]
        # Derivative of eigenvalue wrt matrix - see https://math.stackexchange.com/questions/2588473/derivatives-of-eigenvalues
        loss_del_cov = np.einsum('i,j->ij', min_eigvec, min_eigvec)
        loss_del_d = np.einsum('ij,ijk->k', loss_del_cov, cov_del_d)
    return loss, loss_del_d