# Hopfield classification
import math
import numpy as np

class Neuron:
    def __init__(self, weights, bias, tfuncction):
        self.w = weights
        self.b = bias
        self.tf = tfuncction
        
    def purelin(self, x):
        return x

    def satlins(self, x):
        x = np.where(x<-1, -1, x)
        return np.where(x>1, 1, x)

    def calculate(self, inputs):
        res = np.dot(self.w, inputs) + self.b

        if (self.tf == "SATLINS"):
            res = self.satlins(res)
        else:
            raise Exception('TF not set.')

        return res

w = np.matrix([[0.2, 0, 0], [0, 1.2, 0], [0, 0, 0.2]])
b = np.matrix([0.9, 0, -0.9]).T
hn = Neuron(w, b, "SATLINS")

# test
input_vector = np.matrix(([-1, -1, -1])).T

a_old = hn.calculate(input_vector)
a_new = hn.calculate(a_old)
c = 2
max_loops = 1000

while ((not (a_old == a_new).all) and (c < max_loops)):
    a_old = a_new
    a_new = hn.calculate(a_old)
    c += 1

print(a_new)
print(c)

