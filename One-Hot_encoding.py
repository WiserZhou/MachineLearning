from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
print(encoder.fit_transform([['red'], ['green'], ['blue']]))
'''
[[0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]]
 '''


from keras.utils import to_categorical

print(to_categorical([0, 1, 2]))
'''
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
 '''

import numpy as np

arr = [2, 1, 0]
max = np.max(arr) + 1
print(np.eye(max)[arr])
'''
[[0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]]
'''