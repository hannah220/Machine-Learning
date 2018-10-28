import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep_k(gamma, X, K):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    N = X.shape[0]           # number of samples
    #N = 2
    soft_num = [0] * K
    weights = [0] * K
    means = np.zeros((K, 2))
    covariances = np.zeros((2, 2, K))
    total = [0] * K
    
    for k in range(0, K, 1):
        for n in range(0, N, 1):
            soft_num[k] += gamma[n][k]

    for k in range(0, K, 1):
        weights[k] = soft_num[k] / N

    for k in range(0, K, 1):
        for n in range(0, N, 1):
            means[k] += gamma[n][k] * X[n]
        means[k] = 1/soft_num[k] * means[k]

    for k in range(0, K, 1):
        for n in range(0, N, 1):
            dis = np.matrix(X[n] - means[k])
            total[k] += gamma[n][k] * np.dot(dis.T, dis)
        covariances[:, :, k] = 1/soft_num[k] * total[k]
    
    #logLikelihood = getLogLikelihood(means, weights, covariances, X)
    logLikelihood = 3

    #####Insert your code here for subtask 6c#####
    return weights, means, covariances, logLikelihood



