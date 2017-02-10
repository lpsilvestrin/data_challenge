import numpy as np

from modular_nn import *

Xtr = np.genfromtxt('../data/Xtr.csv', delimiter=',')
Ytr = np.genfromtxt('../data/Ytr.csv', delimiter=',',skip_header=1)
#Xte = np.genfromtxt('data/Xte.csv', delimiter=',')

X=Xtr[:,0:-1]
X_train=X[0:4000,:]
X_test=X[4000:5000,:]
Y_train=Ytr[0:4000,1]
Y_test=Ytr[4000:5000,1]


indptr = range(len(Y_train)+1)
Y_matrix= np.zeros((len(Y_train),10))
for i in xrange(len(Y_train)):
    Y_matrix[i][int(Y_train[i])]=1

topology = [3072,30, 10]
X=X_train
y=Y_test
nn = Neural_Network_modular(topology, 0.01, X[0], y)
nn.forward(X[0])
print nn.output_layer.backward(y[0])
print "\n"
print nn.layers[0].W

nn.backward(y[0])
nn.update()