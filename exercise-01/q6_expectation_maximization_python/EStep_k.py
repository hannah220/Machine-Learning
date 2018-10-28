import numpy as np
import math 
from getLogLikelihood import getLogLikelihood
from getLogLikelihood_k import getLogLikelihood_k
pi = np.pi


def EStep_k(means, covariances, weights, X, K):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    N = X.shape[0]           # number of samples
    numerator = [0] * K
    sub_total = [0] * N
    gamma = np.zeros([N, K])

    for n in range(0, N, 1):
        for k in range(0, K, 1):
            dis = np.matrix(X[n] - means[k])
            inv_cov = np.linalg.inv(covariances[:, :, k])
            det_cov = np.linalg.det(covariances[:, :, k])
            mul = np.linalg.multi_dot([dis, inv_cov, dis.T])
            numerator[k] = weights[k] * (1/(2 * pi * math.sqrt(det_cov))) * math.exp(-1/2 * mul)
            sub_total[n] += weights[k] * (1/(2 * pi * math.sqrt(det_cov))) * math.exp(-1/2 * mul)
        for k in range(0, K, 1):
            gamma[n][k] = numerator[k] / sub_total[n]

    logLikelihood = getLogLikelihood_k(means, weights, covariances, X, K)
    #####Insert your code here for subtask 6b#####
    return [logLikelihood, gamma]
