import numpy as np

def linclass(weight, bias, data):
    # Linear Classifier
    #
    # INPUT:
    # weight      : weights                (dim x 1)
    # bias        : bias term              (scalar)
    # data        : Input to be classified (num_samples x dim)
    #
    # OUTPUT:
    # class_pred       : Predicted class (+-1) values  (num_samples x 1)
    
    N = data.shape[0]
    class_pred = np.zeros([N, 1])
    
    for n in range(N):
        y = np.dot(weight.T, data[n].T)
        print('y is\n', y)
        if y >= 0:
            class_pred[n] = 1
        else:
            class_pred[n] = -1

    #####Insert your code here for subtask 1b#####
    # Perform linear classification i.e. class prediction
    return class_pred


