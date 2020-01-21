# One-input neuron
import math
import random

class Neuron:
    def __init__(self, weight, bias, tfuncction):
        self.w = weight
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

    def calculate(self, i):
        res = i * self.w + self.b

        if (self.tf == "HARDLIM"):
            res = self.hardlim(res)
        if (self.tf == "PURELIN"):
            res = self.purelin(res)
        if (self.tf == "LOGSIG"):
            res = self.logsig(res)

        return res

#test
sn = Neuron(-0.5, 0.1, "LOGSIG")
a = sn.calculate(1.2)
print (a)


