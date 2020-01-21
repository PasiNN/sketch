# Classification problem
import math
import numpy as np
import random

class Neuron:
    def __init__(self, weights, bias, tfuncction):
        self.w = weights
        self.b = bias
        self.tf = tfuncction

    def hardlims(self, x):
        if (x < 0):
            res = -1
        else:
            res = 1
        return res

    def calculate(self, inputs):
        res = np.dot(self.w, inputs) + self.b

        if (self.tf == "HARDLIMS"):
            res = self.hardlims(res)
        else:
            raise Exception('TF not set.')

        return res

# test
sn = Neuron([0, 1, 0], 0, "HARDLIMS")
print (sn.calculate([1, -1, -1]))
print (sn.calculate([1, 1, -1]))
print (sn.calculate([-1, -1, -1]))


