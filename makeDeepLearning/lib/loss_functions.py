import numpy as np

def mean_squared_error(y, t):
    return np.sum((y-t)**2)/2

#print(mean_squared_error(np.array([2,2,0.1,4]),np.array([2,1,0.9,1]))) #5.32

# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#     batch_size = y.shape[0]
#     dlt = 1e-7
#     return -np.sum(t*np.log(y+dlt))/batch_size


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
 # print(cross_entropy_error(np.array([3,2]),np.array([2,1]))) #-2.89037187456
