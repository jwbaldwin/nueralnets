import numpy as np

# sigmoid
def sigmoid(x, deriv=False):
    if(deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output dataset
y = np.array([[0, 0, 1, 1]]).T

'''
3 input - 1 output NN:

0
| \
0 - >  0
| /
0
'''

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3, 1)) - 1

print("y: \n", y)
print("syn0: \n", syn0)

for x in range(100000):

    # forward propagation
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * sigmoid(l1, True)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)
    # print("l1: \n", l1)
    # print("err: \n", l1)
    # print("delta: \n", l1_delta)
    # print("syn0: \n", syn0)

print ("Output After Training:")
print (l1)
