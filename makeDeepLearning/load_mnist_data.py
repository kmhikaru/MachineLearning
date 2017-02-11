import sys, os
sys.path.append('./lib')
import functions as fn
import loss_functions as lf
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

# print(x_train.shape) (60000,784)

train_size = x_train.shape[0] #60000

batch_size = 10

batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(lf.cross_entropy_error(x_batch, x_batch))
# print(lf.mean_squared_error(np.array([3,2]),np.array([2,1])))
