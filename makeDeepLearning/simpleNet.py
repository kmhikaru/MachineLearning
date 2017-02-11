import sys, os
sys.path.append('./lib')
import numpy as np
import functions as fn
import loss_functions as lf

class simpleNet:

    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)

        return loss


net = simpleNet()
print(net.W)
x = np.random.randn(1,2)
print(net.predict(x))
