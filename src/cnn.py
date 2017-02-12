import numpy as np
from conv import Conv_layer
from pool import Pool_layer
from modular_nn import Neural_Network_modular

class Convolutional_nn(object):
    def __init__(self, input_shape, nb_classes):
        conv1 = Conv_layer(5, [3, 32,32], 16)
        conv2 = Conv_layer(5, [16, 14, 14], 64)
        conv3 = Conv_layer(5, [64, 5,5], 256)
        pool1 = Pool_layer(2)
        pool2 = Pool_layer(2)
        nn = Neural_Network_modular([256, 500, nb_classes],0,0)
    
    def forward(self, X):
        x_tmp = X
        y1 = conv1.forward(X)
        a1 = pool1.forward(y1)
        y = conv2.forward(a1)
        a = pool2.forward(y)
        y = conv3.forward(a)
        y_hat = nn.forward(y)
        return y_hat
        
    def backward(self, Y):
    	delta, W = nn.backward(Y)
    	delta=np.dot(delta, (W[0:-1,:]).T)
    	delta = pool.backward(delta)
    	delta = conv.backward(delta)
    	delta = pool.backward(delta)
    	conv.backward(delta)
    
    def update(self):
        nn.update()
        conv.update()
  
    	
   
