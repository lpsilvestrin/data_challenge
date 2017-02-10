import numpy as np
from modular_nn import Neural_Network_modular 

X = np.array(([3,5,4,2,3], [5,1,6,4,7], [10,2,2,3,4], [5, 10, 2, 6, 7]), dtype=float)
y = np.array(([75,1], [82,2], [93,3], [20, 0.5]), dtype=float)

X = X/np.amax(X, axis=0)
y = y/100

topology = [5, 10, 7, 2]


nn = Neural_Network_modular(topology, 0.01, X, y)

batch_size = 2
nn.batch_train(X, y, batch_size)
err = nn.test(X,y)
y0 = nn.forward(X[0])

print err
print y0, y[0]
