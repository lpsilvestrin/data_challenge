from scipy import misc
import numpy as np
import sys

fname = sys.argv[1]

data = np.genfromtxt(fname, delimiter=',')
data = data[:,0:-1]
img1 = data[0,:]
img1 = np.reshape(img1, (32,32,3))

misc.imsave("test.jpg",img1)
