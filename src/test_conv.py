import numpy as np
from scipy import signal
from conv import Conv_layer
from pool import Pool_layer

feature_map = np.array([[[1,5,3], [1,2,9], [9,1,2]],[[7,2,1], [4,5,7], [0,7,4]]])



kerneldim2 = np.array([[1,1],[2,2]])
kerneldim1 = np.array([[1,1],[2,2]])

matrix = feature_map[0,:,:]
kernel = kerneldim1

ksize = kernel.shape[0]
input_shape = [2,3,3]

#nb_features = kernel.shape[2]
c = Conv_layer(ksize, input_shape, nb_features=4)
out = c.forward(feature_map)
print out

print feature_map

print c.backward(out)
c.update(3)

#kernel = np.concatenate((kerneldim1.reshape([4,1]), 
#		kerneldim2.reshape([4,1])))




