import numpy as np
from scipy import sparse


x = [[0,1,2], 
    [3,0,5],
    [6,7,0]]

x = np.array(x)

sA = sparse.csr_matrix(x) 

source_idx, neighbor_idx = sA.nonzero()

print('source_idx: ', source_idx)
print(sA[source_idx, neighbor_idx].A1)



x = np.arange(6).reshape(2, 3)

y = np.arange(2)

print('x shape' ,x.shape)
print(x)
print()
print('y shape before: ', y.shape)
print('y shape after: ', y[:, None].shape)
print(y)

print()
z = x * y[:, None] 

print(z)


