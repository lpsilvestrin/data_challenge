# Define a convolutional layer

from scipy import signal
import numpy as np
import sys

class conv_layer(ksize, stride):
	
	def __init__(self, ksize, stride):
		self.k = zeros([ksize, ksize])
		self.s = stride
		
	def forward(img, size):
		return signal.convolve2d(img, self.k)
		
	def backward():
		return
