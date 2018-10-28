import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from EStep import EStep
from EStep_k import EStep_k
from MStep import MStep
from MStep_k import MStep_k
from regularize_cov import regularize_cov


def estGaussMixEM_k(data, K, n_iters, epsilon):
    # EM algorithm for estimation gaussian mixture mode
    #
    # INPUT:
    # data           : input data, N observations, D dimensional
    # K              : number of mixture components (modes)
    #
    # OUTPUT:
    # weights        : mixture weights - P(j) from lecture
    # means          : means of gaussians
    # covariances    : covariancesariance matrices of gaussians
    # logLikelihood  : log-likelihood of the data given the model

    N = data.shape[0]
    weights = np.ones(K) / K
    covariances = np.ndarray((2, 2, 3))

    kmeans = KMeans(n_clusters = K, n_init = 10).fit(data)
    cluster_idx = kmeans.labels_
    means = kmeans.cluster_centers_

    # Create initial covariance matrices
    for j in range(K):
        data_cluster = data[cluster_idx == j] 
        min_dist = np.inf
        for i in range(K):
            # compute sum of distances in cluster
            dist = np.mean(euclidean_distances(data_cluster, [means[i]], squared=True)) 
            if dist < min_dist:
                min_dist = dist
        covariances[:, :, j] = np.eye(2) * min_dist

    for n in range(n_iters):
        _, gamma = EStep_k(means, covariances, weights, data, K)
        weights, means, covariances, _ = MStep_k(gamma, data, K)

    #####Insert your code here for subtask 6e#####
    return [weights, means, covariances]

