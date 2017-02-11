# ~*~ coding:utf-8 ~*~
import numpy as np

def identity_function(x):
	return x

def sigmoid(x):
	return 1/(1+np.exp(-x))

def softmax(x):
	c = np.max(x)
	x = x - c
	y = np.exp(x)/sum(np.exp(x))
	return y


def numerical_diff(f, x):
	h = 1e-4
	return (f(x+h) - f(x-h)) / (2*h)
# print(softmax(np.array([101,102.100]))) #[ 0.24973989  0.75026011]

def _numerical_gradient_no_batch(f, x):
	h = 1e-9
	grad = np.zeros_like(x)
	for idx in range(x.size):
		# tmp = x[idx]
		f1 = f(x[idx] + h)
		f2 = f(x[idx] - h)
		grad[idx] = (f1-f2)/(2*h)
		# tmp_val = x[idx]
		# x[idx] = tmp_val + h
		# fxh1 = f(x)
		#
		# x[idx] = tmp_val - h
		# fxh2 = f(x)
		#
		# grad[idx] = (fxh1 -fxh2) / (2*h)
		# x[idx] = tmp_val

	return grad
#
# def numerical_gradient(f, X):
# 	if X.ndim == 1:
# 		return _numerical_gradient_no_batch(f, X)
# 	else:
# 		grad = np.zeros_like(X)
#
# 		for idx, x in enumerate(X):
# 			print(idx)
# 			grad[idx] = _numerical_gradient_no_batch(f, x)
#
# 		return grad

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 値を元に戻す
        it.iternext()

    return grad

def quadratic(x):
	return np.sum(x**2)

print(numerical_gradient(quadratic,np.array([2,2])))
