import numpy as np

# max pool layer
class Pool_layer():
    def __init__(self, size):
        self.size = size
        self.X = 0
        self.Z = 0

    # inputed feature maps should have square shape
    def forward(self, X):
        shape = X.shape
        nb_features = shape[0]
        in_size = shape[1]
        self.X = X
        res_size = in_size/self.size
        self.Z = np.zeros([nb_features, res_size, res_size])
        for d in range(nb_features):
            self.Z[d] = self.max_pool1(X[d], res_size, in_size)
        
        return self.Z
    
    # computes de derivative of the max pool w.r.t the input
    def backward(self, next_delta):
        shape = self.X.shape
        delta = np.zeros(shape)
        res_size = shape[1]/self.size
        next_delta = next_delta.reshape(shape[0], res_size, res_size)
        for d in range(shape[0]):
            for i in range(0, shape[1], self.size):
                for j in range(0, shape[2], self.size):
                    c1,c2 = self.max_pool_back(self.X[d],i,j)
                    delta[d,i+c1,j+c2] = next_delta[d,i/self.size,j/self.size]
        return delta
       
    # max pool over one feature map
    # X : input
    # res_size : size of the pooling result
    # in_size: size of the input square matrix
    def max_pool1(self, X, res_size, in_size):
        Z = np.zeros([res_size, res_size])
        i_out = 0
        for i in range(0, in_size, self.size):
            j_out = 0
            for j in range(0, in_size, self.size):
                Z[i_out,j_out] = self.max_pool(X,i,j)
                j_out += 1
            i_out += 1
            
        return Z
    
    
    # max pool over a size x size portion of the input
    def max_pool(self, X, i,j):
        size = self.size
        m = -np.inf
        for k1 in range(size):
            for k2 in range(size):
                if m < X[i+k1,j+k2]:
                    m = X[i+k1,j+k2]
        return m
	
	# give the relative coordinates of the max in a portion of the input
    def max_pool_back(self, X,i,j):
        size = self.size
        m = -np.inf
        c1 = 0
        c2 = 0
        for k1 in range(size):
            for k2 in range(size):
                if m < X[i+k1, j+k2]:
                    m = X[i+k1, j+k2]
                    c1 = k1
                    c2 = k2
        return c1,c2
	    
