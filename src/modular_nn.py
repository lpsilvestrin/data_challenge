import numpy as np
import utils

class Layer(object):
    def __init__(self,input_size, nb_neurons, activation, activation_prime):
        #sizes
        self.nb_neurons= nb_neurons
        self.input_size = input_size
        #activation function and derivative
        self.activation = activation
        self.activation_prime = activation_prime
        #Weights (parameters)
        self.W = np.random.randn(self.input_size, self.nb_neurons)
        #input
        self.x = np.zeros(input_size)
        #output
        self.z = np.zeros(nb_neurons)
        
    def forward(self,x):
    	self.x = x
        self.z=np.dot(self.x, self.W)
        self.a=self.activation(self.z)
        return self.a
        
    def backward(self,next_delta,next_W):
        self.delta=np.dot(next_delta, next_W.T)*self.activation_prime(self.z)
        self.dJdW=np.outer(self.x, self.delta)
        return self.delta, self.W
        
    def update(self, l_rate):
        self.W = self.W - l_rate * self.dJdW


class Output_layer(object):
    def __init__(self,input_size, nb_neurons, activation, activation_prime, error_prime):
        #sizes
        self.nb_neurons= nb_neurons
        self.input_size = input_size
        #Weights (parameters)
        self.W = np.random.randn(self.input_size, self.nb_neurons )
        #activation function and derivative
        self.activation = activation
        self.activation_prime = activation_prime
        self.error_prime = error_prime
        #input
        self.x = np.zeros(input_size)
        #output
        self.a = np.zeros(nb_neurons)
        
        
    def forward(self,x):
    	self.x = x
        self.z=np.dot(self.x, self.W)
        self.a=self.activation(self.z)
        return self.a
       
    # starts back propagation given the target output y and the derivative
    # of the error function
    def backward(self, y):
        self.delta = np.multiply(self.error_prime(y, self.a), self.activation_prime(self.z))
        self.dJdW = np.outer(self.x, self.delta) 
        return self.delta, self.W
    
    def update(self, l_rate):
        self.W = self.W - l_rate * self.dJdW
    
class Neural_Network_modular(object):
    def __init__(self, topology, l_rate, X, Y):
        self.topology=topology
        self.l_rate = l_rate
        self.X = X
        self.Y = Y
        #setting model functions
        self.activation = utils.sigmoid
        self.activation_prime = utils.sigmoid_prime
        self.error = utils.mean_square
        self.error_prime = utils.mean_square_prime
        self.layers = []
        i = 1
        # initialize each layer until the last one (output layer)
        while i < len(topology)-1:
            l = Layer(topology[i-1], topology[i], self.activation, self.activation_prime)
            self.layers.append(l)
            i = i + 1
        self.output_layer = Output_layer(topology[-2], topology[-1], self.activation, self.activation_prime, self.error_prime)
    
    def forward(self, X):
        x_tmp = X
        for i in range(len(self.layers)):
            a = self.layers[i].forward(x_tmp)
            x_tmp = a
        a = self.output_layer.forward(x_tmp)
        return a

	# do the backpropagation to calculate the gradient
	# input : y -- class of the input of the forward fase
    def backward(self, y):
        delta, W = self.output_layer.backward(y)
        i = len(self.layers) - 1
        while i >= 0:
            d, next_w = self.layers[i].backward(delta, W)
            delta = d
            W = next_w
            i -= 1
	
    def update(self):
        self.output_layer.update(self.l_rate)
        for i in range(len(self.layers)):
            self.layers[i].update(self.l_rate)
            
	def train(self):
		return
		
            

