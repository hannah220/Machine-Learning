import numpy as np


def regularize_cov(covariance, epsilon):
    # regularize a covariance matrix, by enforcing a minimum
    # value on its singular values. Explanation see exercise sheet.
    #
    # INPUT:
    #  covariance: matrix
    #  epsilon:    minimum value for singular values
    #
    # OUTPUT:
    # regularized_cov: reconstructed matrix
   
    i = np.array([[1, 0], [0, 1]])
    regularized_cov = covariance + epsilon * i
    #####Insert your code here for subtask 6d#####
    return regularized_cov
