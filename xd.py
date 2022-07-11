import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from kneed import KneeLocator
import igraph
import tensorflow.compat.v1 as tf

x = np.array([[1,0,0,3],[2,0,7,0],[0,11,0,3],[15,0,0,0]])

r, c = x.nonzero()

# print('r: ', r)
# print('c: ', c)


batch_labels = np.random.randint(2, size=(4))
attr_matrix = np.random.rand(4,15)
batch_idx = r 
batch_pprw = x[r,c]
batch_attr = attr_matrix[c]

# print('batch idx shape: ', batch_idx.shape)
# print('batch_pprw: ', batch_pprw.shape)
# print('attr_matrix: ', attr_matrix.shape)
# print('batch_attr: ', batch_attr.shape)

hidden_size = 9
nlayers =2
nc = 2

w_0 = np.random.rand(15,hidden_size)
w_1 = np.random.rand(hidden_size,hidden_size)
w_2 = np.random.rand(hidden_size,nc)

h = batch_attr @ w_0
print('batch_attr @ w_0', h.shape)
h = h @ w_1
print('h @ w_1', h.shape)
h = h @ w_2
print('h @ w_2', h.shape)

logits = h

print('logits: ', logits)


print('batch_pprw[:, None]' ,batch_pprw[:, None])


xd = logits * batch_pprw[:, None]
print('xd: ', xd)


# print('batch_idx[:, None]', batch_idx[:, None])

# weights = np.zeros((4, nc))

# for i, a in enumerate(batch_idx):
#     weights[a] += xd[i]

# print(weights)
# weighted_logits = tf.tensor_scatter_nd_add(tf.zeros((tf.shape(batch_labels)[0], 4)),
#                                                    batch_idx[:, None],
#                                                    logits * batch_pprw[:, None])


y = np.zeros((512, 8710))
z = np.zeros((512, 8710))

print((y*z).shape)



