import numpy as np

from cnn import Convolutional_nn

Xtr = np.genfromtxt('../data/Xtr.csv', delimiter=',')
Ytr = np.genfromtxt('../data/Ytr.csv', delimiter=',',skip_header=1)
#Xte = np.genfromtxt('data/Xte.csv', delimiter=',')

# adding the intercept
X = Xtr[:,0:-1]
X_train=X[0:4000,:]
X_test=X[4000:5000,:]
Y = Ytr[:,1]

indptr = range(len(Y)+1)
Y_matrix= np.zeros((len(Y),10))
for i in xrange(len(Y)):
    Y_matrix[i][int(Y[i])]=1
    
Y_train = Y_matrix[0:4000]
Y_test = Y_matrix[4000:5000]

x0 = X[0].reshape(3,32,32)
classes = 10
l_rate = 0.01
cnn = Convolutional_nn(x0.shape, classes, l_rate)


