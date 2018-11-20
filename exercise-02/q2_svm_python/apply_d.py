import numpy as np
from parameters import parameters
from svmkern import svmkern
from kern import kern

C, C2, norm = parameters()

test = {}
train = {}
## Load the training data
train.update({'data': np.loadtxt('lc_train_data.dat').T})
train.update({'label': np.loadtxt('lc_train_label.dat').T})
test.update({'data': np.loadtxt('lc_test_data.dat').T})
test.update({'label': np.loadtxt('lc_test_label.dat').T})

# Kernel SVM
[alpha, sv, b, _, slack]  = svmkern(train['data'], train['label'], C2, norm)

# Run the classifier on the test data
# Unlike the linear SVM (which only needs the computed weight w and bias b),
# the kernel SVM requires a subset of the training dataset (the support vectors)
# to perform classification on a test dataset.
result = np.sign((kern(test['data'], train['data'][:,sv],norm).T).dot(alpha[sv].T*train['label'][sv])+ b)

# Accuracy on test data
accuracy = len(result[result==test['label']])/len(test['label'])
print('Accuracy of kernel SVM with C={0} and norm={1}: {2}\n'.format(C2, norm, accuracy))
