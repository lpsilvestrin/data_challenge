import numpy as np
from conv import Conv_layer
from pool import Pool_layer
from modular_nn import Neural_Network_modular

class Convolutional_nn(object):
    def __init__(self, input_shape, nb_classes, l_rate):
        self.l_rate = l_rate
        self.conv1 = Conv_layer(5, [3, 32,32], 16)
        self.conv2 = Conv_layer(5, [16, 14, 14], 64)
        self.conv3 = Conv_layer(5, [64, 5,5], 256)
        self.pool1 = Pool_layer(2)
        self.pool2 = Pool_layer(2)
        self.nn = Neural_Network_modular([256, 500, nb_classes],l_rate, 0,0)
    
    def forward(self, X):
        x_tmp = X
        y1 = self.conv1.forward(X)
        a1 = self.pool1.forward(y1)
        y = self.conv2.forward(a1)
        a = self.pool2.forward(y)
        y = self.conv3.forward(a)
        y_hat = self.nn.forward(y.reshape(256))
        return y_hat
        
    def backward(self, Y):
    	delta, W = self.nn.backward(Y)
    	delta=np.dot(delta, (W[0:-1,:]).T)
    	delta = self.conv3.backward(delta.reshape(256,1,1))
    	delta = self.pool2.backward(delta)
    	delta = self.conv2.backward(delta)
    	delta = self.pool1.backward(delta)
    	delta = self.conv1.backward(delta)
    
    def update(self):
        self.nn.update()
        self.conv1.update(self.l_rate)
        self.conv2.update(self.l_rate)
        self.conv3.update(self.l_rate)
  
    	
   
