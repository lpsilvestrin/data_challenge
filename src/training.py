import numpy as np

from modular_nn import Neural_Network_modular
from utils import create_back_up
from utils import upload_back_up

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

topology = [3072, 200, 10]

nn = Neural_Network_modular(topology, 0.1, X_train[0], Y_train)
batch_size = 100
nn.batch_train(X_train, Y_train, batch_size)
err = nn.test(X_train, Y_train)
y_hat0 = nn.forward(X_train[0])
print "error: ", err
print "\n"
print y_hat0
print Y_train[0]

