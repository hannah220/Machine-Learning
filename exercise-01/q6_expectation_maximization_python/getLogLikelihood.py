import numpy as np
import math
pi = np.pi

def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    N = X.shape[0]           # number of samples
    K = 3                    # number of Gaussions
    sub_total = [0] * N
    logLikelihood = 0

    for n in range(0, N, 1):
        for k in range(0, K, 1):
            dis = np.matrix(X[n] - means[k])
            inv_cov = np.linalg.inv(covariances[:, :, k])
            det_cov = np.linalg.det(covariances[:, :, k])
            mul = np.linalg.multi_dot([dis, inv_cov, dis.T])
            sub_total[n] += weights[k] * (1/(2 * pi * math.sqrt(det_cov))) * math.exp(-1/2 * mul)

        logLikelihood += np.log(sub_total[n])

    #####Insert your code here for subtask 6a#####
    return logLikelihood

