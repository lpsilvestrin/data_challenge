import numpy as np

from modular_nn import Neural_Network_modular

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

topology = [3072, 200, 10]
X_train
Y=Y_train
nn = Neural_Network_modular(topology, 0.1, X_train[0], Y_train)
batch_size = 20
nn.batch_train(X_train, Y_train, batch_size)
err = nn.test(X_train, Y_train)
y_hat0 = nn.forward(X_train[0])
print "error: ", err
print "\n"
print y_hat0
print Y_train[0]

