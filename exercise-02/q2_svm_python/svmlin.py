import numpy as np
#import os # might need to add path to mingw-w64/bin for cvxopt to work
#os.environ["PATH"] += os.pathsep + ...
from cvxopt import solvers
from cvxopt import matrix

def svmlin(X, t, C, ex):
    # Linear SVM Classifier
    #
    # INPUT:
    # X        : the dataset                  (dim x num_samples)????
    # t        : labeling                     (num_samples x 1)
    # C        : penalty factor for slack variables (scalar)
    #
    # OUTPUT:
    # alpha    : output of quadprog function  (num_samples x 1)
    # sv       : support vectors (boolean)    (1 x num_samples)
    # w        : parameters of the classifier (1 x dim)
    # b        : bias of the classifier       (scalar)
    # result   : result of classification     (1 x num_samples)
    # slack    : points inside the margin (boolean)   (1 x num_samples)
   
    print('X is\n', X.shape)
    N = X.shape[0]
    dim = X.shape[1]
    print('N is\n', N)
    print('dim is\n', dim)
    print('t is \n', t.shape)
#    a = np.zeros([N, 1])    # N x 1
    H = np.zeros([N, N])    # N x N
#    P = np.zeros([N, N])    # N x N
    c = C * np.ones([N, 1])    # N x 1
    one = np.ones([N, 1])   # N x 1
    zero = np.zeros([N, 1]) # N x 1
    h0 = np.zeros([2*N, 1])    # 2N x 1
    w = np.zeros([1, dim])  # 1 x dim
    
    for i in range(N):
        for j in range(N):
            H[i, j] = t[i] * t[j] * np.dot(X[j, :].T, X[i, :])
    print('H is\n', H.shape)

    G = matrix(np.vstack([-np.eye(N), np.eye(N)]))
    q = matrix((-1.0)* np.ones(N))
    P = matrix(H)
    t = np.matrix(t)
    A = matrix(t)
    h0[0:N, 0] = one[0:N, 0]    
    h0[N:2*N, 0] = c[0:N, 0]     # Quadratic programming
    h = matrix(h0)
    b0 = matrix(0.0)

    a = solvers.qp(P, q, G, h, A, b0)
    alpha = a
    print('alpha is\n', alpha['x'])

    # Calculate hyperplane and bias
    t = t.T
    for n in range(N):
        for d in range(dim):
            w[0, d] += alpha['x'][n] * t[n] * X[n, d]

    # Find out support vectors and put them in support vector matrix 
    sv = np.zeros([1, N])
    support_num = 0
    support_t = np.empty([0, 1])
    support_a = np.empty([0, 1])
    support_X = np.empty([0, dim], float)
    for n in range (N):
        if alpha['x'][n] > 0:
            # add this point to support vector
            support_num += 1
            support_t = np.vstack([support_t, t[n]])
            support_a = np.vstack([support_a, alpha['x'][n]])
            support_X = np.vstack([support_X, np.matrix(X[n, :])])
    print('support_num is\n', support_num)
    
    # Calculate bias b
    sub = np.zeros([1, support_num])
    b = 0
    for n in range (support_num):
        for m in range (support_num):
            sub[0, n] += alpha['x'][m] * t[m] * np.dot(support_X[m, :], support_X[n, :].T)
        #print('sub is\n', sub[0, n])
        b += 1/support_num * (t[n] - sub[0, n])
    
    # Classify xi
    classifier = np.zeros([1, N])
    xi = np.zeros([1, N])
    miss = 0
    for n in range(N):
        classifier[0, n] = np.dot(w, X[n, :].T) + b
        if classifier[0, n] == 0:
            # this point is on the decision boundary
            xi[0, n] = 1
        elif t[n] * classifier[0, n] >= 1:
            # this point is on the correct side of the margin
            xi[0, n] = 0
        else:
            # this point is missclassfied (here xi does not exist)
            xi[0, n] = abs(t[n] - classifier[0, n])
            miss+=1
            #print('xi on the missclassfied point is\n', xi[0, n])
    print('miss is\n', miss)

    # Find out slack points (points inside the margin)
    slack = np.zeros([1, N])
    for n in range(N):
        if 0 < xi[0, n] < 1:
            # this point is slack point
            slack[0, n] = True
        else:
            slack[0, n] = False
    slack = slack.astype(int)
    print('slack is\n', slack)

    # List support vectors
    for n in range(N):
        if t[n] * classifier[0, n] -1 + xi[0, n] == 0 or alpha['x'][n] > 0:
            # this point is support vector
            sv[0, n] = True
        else:
            sv[0, n] = False
    sv = sv.astype(int)
    print('sv is\n', sv)
    
    # Classify datapoints
    result = np.zeros([1, N])
    for n in range(N):
        if classifier[0, n] >= 1 - xi[0, n]:
            result[0, n] = 1
        elif classifier[0, n] <= -1 + xi[0, n]:
            result[0, n] = -1
    print('target is\n', t)
    print('result is\n', result)

    #####Insert your code here for subtask 2a#####
    return alpha, sv, w, b, result, slack



