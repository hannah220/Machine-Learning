import numpy as np

def leastSquares(data, label):
    # Sum of squared error shoud be minimized
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    #
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)
   
    dim = 2
    e = 0   #sum of squared error
    N = data.shape[0]
    weight = np.zeros([dim, 1])
    t = label

    x = np.zeros([N, 3])
    x[:, 0] = 1
    for n in range(N):
        x[n, 1:3] = data[n]

    w = np.dot(np.linalg.pinv(x), t)
    print('w is: \n', w) 
    print('w shape is: \n', w.shape)
    weight[0] = w[1]
    weight[1] = w[2]
    bias = w[0]

    print('weight is: \n', weight) 

    #for n in range(N):
    #    e += 1/2 * pow((np.dot(weights.T, data[n, :]) + bias - label[n]), 2)

    #####Insert your code here for subtask 1a#####
    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)
    return weight, bias
