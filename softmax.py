import numpy as np

def softmax(xs):
    return np.exp(xs) / sum(np.exp(xs))

xs = np.array([-1, 0, 3, 5])
print(softmax(xs)) # [0.0021657, 0.00588697, 0.11824302, 0.87370431]