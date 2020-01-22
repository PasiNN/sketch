# Linearly separable sets with linear decision boundary
import math
import matplotlib.pyplot as plt 
import numpy as np
from random import random

class LinearlySeparableSets:
    def __init__(self, samples):
        self.x0 = np.zeros(samples)
        self.y0 = np.zeros(samples)
        self.x1 = np.zeros(samples)
        self.y1 = np.zeros(samples)
        
        # rotation matrix
        theta = random() * 2 * np.pi
        R = np.matrix([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
        # translation after rotation
        offset_x = random() - 0.5
        offset_y = random() - 0.5
        
        for i in range(samples):
            x = random() * 2 - 1
            y = random() - 1
            res = np.dot(R, np.matrix([x, y]).T)
            self.x0[i] = res[0] + offset_x
            self.y0[i] = res[1] + offset_y

        for i in range(samples):
            x = random() * 2 - 1
            y = random()
            res = np.dot(R, np.matrix([x, y]).T)
            self.x1[i] = res[0] + offset_x
            self.y1[i] = res[1] + offset_y


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

    def calculate(self, inputs):
        res = np.dot(self.w, inputs) + self.b

        if (self.tf == "HARDLIM"):
            res = self.hardlim(res)
        else:
            raise Exception('TF not set.')

        return res


def draw_decision_boundary(w, b):
    x = np.linspace(-2,2,10)
    y = ((-w[0,0]/w[0,1]) * x)  - (b / w[0,1])
    plt.plot(x,y)

def test_solution(w, b, lss):
    tn = Neuron(w, b, "HARDLIM")

    for (x,y) in zip(lss.x0, lss.y0):
        ti = np.matrix([x, y]).T
        if (tn.calculate(ti) != 0):
            return False
    for (x,y) in zip(lss.x1, lss.y1):
        ti = np.matrix([x, y]).T
        if (tn.calculate(ti) != 1):
            return False
    return True


# test
lss = LinearlySeparableSets(2)

plt.xlabel('x') 
plt.ylabel('y') 
plt.xlim(-2,2) 
plt.ylim(-2,2) 
plt.axis('square')
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.title('Plot') 

plt.plot(lss.x0, lss.y0, 'bo') 
plt.plot(lss.x1, lss.y1, 'r+') 

# random decision boundary for testing
w = np.matrix([random()*2-1, random()*2-1])
b = random()*2-1
plt.plot(w[0,0], w[0,1], 'g^') 
draw_decision_boundary(w, b)
plt.show() 

print(test_solution(w, b, lss))

