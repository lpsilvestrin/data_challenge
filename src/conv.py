# Define a convolutional layer

from scipy import signal
import numpy as np
import sys
from pool import Pool_layer

class Conv_layer():
	
    def __init__(self, ksize, input_shape, nb_features, stride=0):
        self.ksize = ksize
        self.input_shape = input_shape
        self.X = np.zeros(input_shape)
        nb_channels = input_shape[0]
        # all kernels's weights represented in matrix format
        self.W = np.random.randn((ksize**2)*nb_channels, nb_features)
        # output
        out_size = input_shape[1]-self.ksize + 1
        self.Z = np.zeros([nb_features, out_size])
        self.s = stride
        self.nb_features = nb_features
        #gradient of weights in kernel format
        self.dJdW = np.zeros([nb_features, nb_channels, ksize,ksize])

    def forward(self, X):
        self.X = X
        mat = self.img2conv_mat(X)
        res_size = self.input_shape[1]-self.ksize+1
        return (np.dot(mat,self.W)).reshape([self.nb_features,res_size,res_size])
		
    def backward(self, next_delta):
        shape = next_delta.shape
        nb_features = self.dJdW.shape[0]
        input_channels = self.dJdW.shape[1]
        for d in range(nb_features):
            for i in range(input_channels):
                # convolution w/ flipped kernel
                self.dJdW[d,i] += signal.convolve2d(self.X[i], next_delta[d],mode='valid')
        delta = np.zeros(self.input_shape)
        self.mat2kernel()
        for k in range(input_channels):
            for f in range(nb_features):
                delta[k] += signal.convolve2d(self.W[f,k], next_delta[f])
        self.kernel2mat()
        return delta
        
    def update(self, l_rate):
        self.mat2kernel()
        self.W = self.W - l_rate * self.dJdW
        self.kernel2mat()
		
    # compute the imput matrix of the convolution given
    # a single input feature map
    # X feature map
    def img2conv(self, X):
        input_shape = X.shape
        N = input_shape[0]
        # resulting features' width
        res_w = N - self.ksize + 1
        # input matrix fomat
        mat_w = (self.ksize**2)
        mat_h = res_w**2
        X_mat = np.zeros([mat_h, mat_w])
        
        # building the matrix:
        
        mat_i = 0
        for i in range(N-self.ksize+1):
            for j in range(N-self.ksize+1):
                mat_j = 0
                for k1 in range(self.ksize):
                    for k2 in range(self.ksize):
                        X_mat[mat_i, mat_j] = X[i+k1, j+k2]
                        mat_j += 1
                mat_i += 1 
                
        return X_mat
    
    # compute the input matrix for the convolution given
    # N input feature maps
    # X feature maps
    def img2conv_mat(self, X):
        # number of feature maps
        nb_features = X.shape[0]
        # calculate first feature map
        X_mat = self.img2conv(X[0,:,:])
        for d in range(1,nb_features):
            X_mat = np.concatenate((X_mat, self.img2conv(X[d,:,:])), axis=1)
            
        return X_mat
    
    #change from kernel to matrix representation of the weights
    def kernel2mat(self):
        ch = self.input_shape[0]
        f = self.nb_features
        k = self.ksize
        self.W = (self.W.reshape([f, ch*k*k])).T
        
    #change from matrix to kernel representation of weights
    def mat2kernel(self):
        ch = self.input_shape[0]
        f = self.nb_features
        k = self.ksize
        self.W = (self.W.T).reshape([f, ch, k, k])
        
