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


def create_back_up(nn):
    for i in xrange(len(nn.layers)):
        name=str(i)+'.out'
        np.savetxt(name,nn.layers[i].W, delimiter=',')
    print "output layer ",i
    print nn.output_layer.W.shape
    np.savetxt("output.out",nn.output_layer.W, delimiter=',')
    print "backup created"
    
def upload_back_up(nn):
    for i in xrange(len(nn.layers)):
        name=str(i)+'.out'
        nn.layers[i].W=np.genfromtxt(name, delimiter=',')
    nn.output_layer.W=np.genfromtxt("output.out", delimiter=',')
    print "backup uploaded"
