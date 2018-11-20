import numpy as np
import matplotlib.pyplot as plt
from parameters import parameters
from svmlin import svmlin

C, C2, norm = parameters()

test = {}
train = {}
## Load the training data
train.update({'data': np.loadtxt('lc_train_data.dat')})
train.update({'label': np.loadtxt('lc_train_label.dat')})
test.update({'data': np.loadtxt('lc_test_data.dat')})
test.update({'label': np.loadtxt('lc_test_label.dat')})

## Add outlier to training data
print('Adding outliers...')
train['data'] = np.append(train['data'], [[1.5, -0.4],[1.45, -0.35]], axis = 0)
train['label'] = np.append(train['label'], [[-1],[-1]])

## Train the SVM
print('Training the SVM')
alpha, sv, w, b, result, slack  = svmlin(train['data'], train['label'], C, 'a')
print('Number of SV: {0}\n'.format(sum(sv)))

# Accuracy on train data
accuracy = len(result[np.sign(result)==train['label']])/len(train['label'])
print('Accuracy on train data with C = {0}: \t {1}'.format(C, accuracy))

## Plotting
#xmax = 1.5  #xmax = max(train['data'][0])
#xmin = -0.5 #xmin = min(train['data'][0)
#ymax = 1.5  #ymax = max(train['data'][1])
#ymin = -0.5 #ymin = min(train['data'][1])
xmax = 3  #xmax = max(train['data'][0])
xmin = -3 #xmin = min(train['data'][0)
ymax = 3  #ymax = max(train['data'][1])
ymin = -3 #ymin = min(train['data'][1])

xmargin = (xmax-xmin)/10
ymargin = (ymax-ymin)/10

plt.subplot()
plt.title('Train Set and learned SVM model')
plt.scatter(train['data'][train['label']==1][:,0], train['data'][train['label']==1][:,1], c='b', marker = (8,2,0), linewidth = 0.5) # blue class=1
plt.scatter(train['data'][train['label']==-1][:,0], train['data'][train['label']==-1][:,1], c='r', marker = (8,2,0), linewidth = 0.5) # red class=-1
# Plot the support vectors
plt.scatter(train['data'][sv][:,0], train['data'][sv][:,1], facecolors='none', edgecolors='limegreen', marker = 'o', linewidth = 1.5)
plt.xlim(xmin-xmargin, xmax+xmargin)
plt.ylim(ymin-ymargin, ymax+ymargin)

# Plot the slack points
#if sum(slack) > 0:
#plt.plot(train['data'][slack][:,0], train['data'][sv][:,1], markerfacecolor='none', markeredgecolor='y', marker = 'o')

print('w[0, 0] is\n', w[0, 0])
print('w[0, 1] is\n', w[0, 1])
print('b is\n', b)

#x = np.arange(-xmax,xmax,0.001)
x = np.arange(-xmax,xmax,0.1)
x = np.matrix(x)
y = -(w[0, 0] *x + b)/w[0, 1]
print('x.shape is\n', x.shape)
print('x is\n', x)
print('y.shape is\n', y.shape)
print('y is\n', y)
pos = -(w[0, 0] *x + b + 1)/w[0, 1]
neg = -(w[0, 0] *x + b - 1)/w[0, 1]

plt.plot(x, y)
plt.plot(x, neg, 'b-', linewidth = 0.75)
plt.plot(x, pos, 'r-', linewidth = 0.75)
plt.show()

### Classify test data
#resultv = np.sign(test['data'].dot(w) + b)
#
## Accuracy on test data
#accuracy = len(resultv[resultv==test['label']])/len(test['label'])
#print('Accuracy on test data with C = {0}: \t {1}\n'.format(C, accuracy))
#
## Plot the results on test set
#plt.subplot()
#plt.title('Test Set')
#correct = 0    
#incorrect = 0
#
#plt.plot(x, y, 'k-', linewidth = 0.75)  # class border
#plt.plot(x, neg, 'b-', linewidth = 0.75)  # blue line
#plt.plot(x, pos, 'r-', linewidth = 0.75)  # red line
#
#plt.scatter(test['data'][test['label']==1][:,0], test['data'][test['label']==1][:,1], c='b', marker = (8,2,0), linewidth = 0.5) # blue class=1
#plt.scatter(test['data'][test['label']==-1][:,0], test['data'][test['label']==-1][:,1], c='r', marker = (8,2,0), linewidth = 0.5) # red class=-1
#plt.scatter(test['data'][test['label']!=resultv][:,0], test['data'][test['label']!=resultv][:,1], facecolors='none', edgecolors='k', marker = 'o') # incorrectly classified
#plt.xlim(xmin-xmargin, xmax+xmargin)
#plt.ylim(ymin-ymargin, ymax+ymargin)
#plt.show()
