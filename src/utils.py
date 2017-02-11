import numpy as np

def sigmoid(z):
        #Apply sigmoid activation function to scalar, vector, or matrix
	return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    #Derivative of sigmoid function
    return np.exp(-z)/((1+np.exp(-z))**2)
   

def mean_square(y, y_hat):
	return 0.5*sum((y-y_hat)**2)
	
def mean_square_prime(y, y_hat):
	return -(y - y_hat)
	
def accuracy(y, y_hat):
	acc = 0;
	if y == y_hat:
		acc += 1;
	return acc
