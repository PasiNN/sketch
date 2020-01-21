# Two-input neuron
import math
import numpy as np
import random

class Neuron:
    def __init__(self, weights, bias, tfuncction):
        self.w = weights
        self.b = bias
        self.tf = tfuncction

    def hardlim(self, x):
        if (x < 0):
            res = 0
        else:
            res = 1
        return res

    def purelin(self, x):
        return x

    def logsig(self, x):
       res = 1 / (1 + math.exp(-x))
       return res

    def calculate(self, inputs):
        res = np.dot(self.w, inputs) + self.b

        if (self.tf == "HARDLIM"):
            res = self.hardlim(res)
        elif (self.tf == "PURELIN"):
            res = self.purelin(res)
        elif (self.tf == "LOGSIG"):
            res = self.logsig(res)
        else:
            raise Exception('TF not set.')

        return res

# test
sn = Neuron([1.1, -0.8], 0.5, "PURELIN")
a = sn.calculate([-1.3, 0.4])
print (a)


