import numpy as np
from kern import kern
import cvxopt

def svmkern(X, t, C, p):
    # Non-Linear SVM Classifier
    #
    # INPUT:
    # X             : the dataset                        (dim x num_samples)
    # t             : labeling                           (1 x num_samples)
    # C             : penalty factor the slack variables (scalar)
    # p             : order of the polynom               (scalar)
    #
    # OUTPUT:
    # sv            : support vectors (boolean)          (1 x num_samples)
    # b             : bias of the classifier             (scalar)
    # slack         : points inside the margin (boolean) (1 x num_samples)

    #####Insert your code here for subtask 2d#####
    # we want to get alpha_n >= 0 for all n : positive Lagrange multipliers
    # maximize svm dual formulation, which is quadratic programming problem
    return alpha, sv, b, result, slack
