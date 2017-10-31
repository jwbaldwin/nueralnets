#
# Imports
#
import numpy as np

class BackPropogationNetwork:
    ''' A back-propagation network '''

    #
    # Class members
    #
    layerCount = 0
    shape = None
    weights = []

    #
    # Class methods
    #
    def __init__ (self, layerSize):
        ''' Initialize the network '''

        # Layer info
        self.layerCount = len(layerSize) - 1
        self.shape = layerSize

        # Input/Output data from last run
        self._layerInput = []
        self._layerOutput = []

        # Create the weight arrays
        for (l1, l2) in zip(layerSize[:-1], layerSize[1:]):
            # Bias node means +1 to leading layer
            self.weights.append(np.random.normal(scale=0.1, size = (l2, l1+1)))

    #
    # Run method
    #
    def Run(self, input):
        ''' Run the network based on the input data '''

        lnCases = input.shape[0]

        # Clear out the previous intermediate value lists
        self._layerInput = []
        self._layerOutput = []

        # Run it!
        for index in range(self.layerCount):
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, lnCases])]))

            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1, lnCases])]))

            self._layerInput.append(layerInput)
            self._layerOutput.append(self.sigmoid(layerInput))

        return self._layerOutput[-1].T

    # Transfer functions
    def sigmoid(self, x, Deriv=False):
        if not Deriv:
            return 1/ (1 + np.exp(-x))
        else:
            out = self.sigmoid(x)
            return out*(1 - out)

#
# If run as a script, create a test object
#
if __name__ == "__main__":
    nn = BackPropogationNetwork((2, 2, 2))
    print(nn.shape)
    print(nn.weights)

    lvInput = np.array([[0,0], [1,1], [-1, 0.5]])
    lvOutput = nn.Run(lvInput)

    print("Input:  {0}\nOutput: {1}".format(lvInput, lvOutput))
