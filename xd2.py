import numpy as np


test = np.array([1,2,3,4,5,6,7,8,9])

current_idx = 6 #(number 7)

interval = 4

print(test[current_idx-interval:current_idx+interval])
