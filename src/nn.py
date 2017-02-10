import numpy as np
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)
X = X/np.amax(X, axis=0)
y = y/100
Lambda = 0.0001
class Neural_Network(object):
    def __init__(self, Lambda=0):        
        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        #Regularization Parameter:
        self.Lambda = Lambda
        
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        cd 
    def forward(self, X):
        #Propagate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
    @staticmethod
    def sigmoid(z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    @staticmethod
    def sigmoidPrime(z):
        #Derivative of sigmoid function
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        #We don't want cost to increase with the number of examples, so normalize by dividing the error term by number of examples(X.shape[0])
        J = 0.5*sum((y-self.yHat)**2)
        return J
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2) 
        return dJdW1, dJdW2
    
    def train(X, y):
    	 = size(X)
    		dW1, dW2 = self.costFunctionPrime(X[i], y[i])
    		self.W1 = self.W1 - dW1
    		self.W2 = self.W2 - dW2
    		
    
NN =Neural_Network()
yHat=NN.forward(X)
cost1 = NN.costFunction(X,y)
dJdW1, dJdW2 = NN.costFunctionPrime(X,y)
print dJdW1
