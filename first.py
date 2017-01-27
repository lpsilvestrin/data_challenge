import numpy as np
Xte = np.genfromtxt('data/Xte.csv', delimiter=',')
Ytr = np.genfromtxt('data/Ytr.csv', delimiter=',').astype(int)
Xtr = np.genfromtxt('data/Xtr.csv', delimiter=',')

class NearestNeighbor(object):
    def __init__(self):
        pass
    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y
    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        #print num_test
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        # loop over all test rows
        print "begin prediction"
        for i in xrange(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value 
	    #print i
            #print np.abs(self.Xtr - X[i,:]) #works fine
            """the problem is in np.sum"""
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            #min_index = np.argmin(distances) # get the index with smallest distance
            min_indexes = np.argsort(distances)[0:5] 
            #print min_index
            #print self.ytr[min_index]
            #Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
	    ytr_min = np.take(self.ytr, min_indexes)
	    Ypred[i] = np.argmax(np.bincount(ytr_min)) # predict the label of the nearest example

        return Ypred


#np.set_printoptions(threshold='nan')
X=Xtr[:,0:-1]
X_train=X[0:4000,:]
X_test=X[4000:5000,:]
Y_train=Ytr[0:4000,1]
Y_test=Ytr[4000:5000,1]
print Y_test
nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(X_train, Y_train) # train the classifier on the training images and labels
Yte_predict = nn.predict(X_test) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print "end prediction"
print 'accuracy: %f' % ( np.mean(Yte_predict == Y_test) )
