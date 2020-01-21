# Hamming classification
import math
import numpy as np

class Neuron:
    def __init__(self, weights, bias, tfuncction):
        self.w = weights
        self.b = bias
        self.tf = tfuncction

    def poslin(self, x):
        return np.where(x<0, 0, x)
        
    def purelin(self, x):
        return x

    def calculate(self, inputs):
        res = np.dot(self.w, inputs) + self.b

        if (self.tf == "POSLIN"):
            res = self.poslin(res)
        elif (self.tf == "PURELIN"):
            res = self.purelin(res)
        else:
            raise Exception('TF not set.')

        return res

prototype1 = np.array([1, -1, -1])
prototype2 = np.array([1, 1, -1])

w1 = np.matrix([prototype1, prototype2])
b1 = np.matrix([len(prototype1), len(prototype1)]).T
forward_layer = Neuron(w1, b1, "PURELIN")

classes = 2
delta = 0
epsilon = 1 / (classes + delta)
w2 = np.matrix([[1, -epsilon], [-epsilon, 1]])
b2 = np.matrix([0, 0]).T
recurrent_layer = Neuron(w2, b2, "POSLIN")


# test
input_vector = np.matrix(([-1, -1, -1])).T

a = forward_layer.calculate(input_vector)

a_old = recurrent_layer.calculate(a)
a_new = recurrent_layer.calculate(a_old)
c = 2
max_loops = 1000

while ((not (a_old == a_new).all) and (c < max_loops)):
    a_old = a_new
    a_new = recurrent_layer.calculate(a_old)
    c += 1

print(a_new)
print(c)

