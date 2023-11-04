import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv') 
#Each phot is a 28x28 photo so it has 784 pixels
#first transpose it so each column represents a picture and the first row contains the labels
# [ [---x1---,]         [ -  -  -    -]
#   [---x2---,]           -  -  -    -
#   [---x3---,]           -  -  -    -
#       .                 x1 x2 x3...xn
#       .                 -  -  -    -
#       .                 -  -  -    -
#   [---xn---,]]        [ -  -  -    -] 

data = np.array(data) # converts our data into an numpy array instead of pandas dataframe
m, n = data.shape # gives us the dimensions of our data (m = # of arrays, n = # of elements in each array = 785)
np.random.shuffle(data) # randomly shuffles our data

data_dev = data[0:1000].T # takes the first 1000 rows to use as testing data and transposes the matrix (n x m instead of m x n)
Y_dev = data_dev[0] # this is all of the labels for each image (1, 5, 2, 9, 3, 5, etc)
X_dev = data_dev[1:n] # this is all of the pixel data for each image (each row no longer corresponds to an image after we transposed. Now, each column is an example instead)
X_dev = X_dev / 255 # this gives us a "brightness" level for each pixel that is between 0 and 1 instead of between 0 and 255

data_train = data[1000:m].T # training data (this is significantly more data than the testing data)
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255

def init_params(): # initializes the values for our hidden layer
    W1 = np.random.rand(10, 784) - 0.5 # this is our weight for our hidden layer. This function generates random number between -.5 and .5 for each element of this new 10 x 784 array (we use 10 because we have 10 neurons in our hidden layer)
    b1 = np.random.rand(10, 1) - 0.5 # this is our bias for our hidden layer.
    W2 = np.random.rand(10, 10) - 0.5 # this is our weight for our output layer. The reason this is 10 x 10 instead of 10 x 784 is because our connection to this layer is coming from the hidden layer instead of the input layer. 
                                      # the hidden layer only has 10 units/neurons as opposed to the input layer that has 784.
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0) # goes through each element in our Z array, returning 0 if Z < 0 or Z if Z > 0. 

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z)) # this is the activation function that we use for our output layer. Remember, our output layer is a 10 x 1 array where each element is corresponding to a probability for each number 0-9. 
    return (A)                     # So our probability needs to be a decimal between 0 and 1 for each element in this array (graphically, this function looks very similar to the sigmoid activation function but tends to work better).

def forward_prop(W1, b1, W2, b2, X): # X is our input layer
    Z1 = W1.dot(X) + b1 # we take the dot product of our input layer and W1 and add our bias to it. This is effectively just linear regression at this point until we apply our activation function later.
    A1 = ReLU(Z1) # applies ReLU activation function to Z1
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2) # applies softmax activation function to Z2
    return Z1, A1, Z2, A2

def one_hot(Y): # this is how we encode the correct label, simply setting the correct label to 1 and everything else to zero will accomplish this.
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) # creates array of zeroes. Y.size = the number of examples we have. Y.max() gives us the maximum label (which is 9) and we add 1 to it to get the number of labels we have (0-9 is 10 different labels). Pretty sure we can just hardcode this as 10.
    one_hot_Y[np.arange(Y.size), Y] = 1 # sets the correct label to 1, leaving every other label as zero
    one_hot_Y = one_hot_Y.T # transposes matrix
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0 # the ReLU function is a linear, piecewise function where the slope of the ReLU function is 1 if x > 0 and 0 if x < 0. Because of this, the derivative when x > 0 is just the derivative of 1*x which is 1 and the derivative when x < 0 is just 0. 
                 # The reason this return statement works is because the boolean values will be converted to numberic values (True = 1, False = 0) when we use them in our calculation.

def back_prop(Z1, A1, Z2, A2, W2, X, Y): # this is where we calculate our error and set new values for our weights and biases. Y is the LABEL for our prediction whereas A2 is essentially the confidence for each label. This makes sense considering how functions work, you input an X value and receive a Y value as output.
    m = Y.size # number of examples
    one_hot_Y = one_hot(Y) # Ex: if Y = 4, this returns [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    dZ2 = A2 - one_hot_Y # 
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha): # alpha is our learning rate. This will decrease over time as we refine our results.
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2    
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()  
    print(W1)
    print('here')
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X) 
#         print(Z1)
#         print(A1)
#         print(Z2)
#         print(A2)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2

W1, b1 , W2, b2 = gradient_descent(X_train, Y_train, 500, 0.1)

