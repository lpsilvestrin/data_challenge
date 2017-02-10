import numpy as np
from modular_nn import Neural_Network_modular 

X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

X = X/np.amax(X, axis=0)
y = y/100

topology = [2, 3, 1]


nn = Neural_Network_modular(topology, 0.01, X, y)

print nn.layers[0].nb_neurons
print nn.output_layer.W

y_hat = nn.forward(X[0])
print(y_hat)
