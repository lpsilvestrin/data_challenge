import numpy as np

#activation function
def tan(a):
	return (np.exp(a) - np.exp(-a))/(np.exp(a) + np.exp(-a))

class layer:
	def __init__(self, size, weights):
		self.N = size
		self.weights = weights

	def forward(self, x:
		self.input = x
		output = np.dot(self.weights, x)
		return output
			
	def backward(self, ):
		
		
class fcl():



