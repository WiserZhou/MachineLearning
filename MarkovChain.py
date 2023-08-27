import numpy as np

matrix = np.array([[2/9, 5/9, 2/9],
                   [5/13, 6/13, 2/13],
                   [1/3, 1/2, 1/6]], dtype=float)
vector1 = np.array([[5/13, 6/13, 2/13]], dtype=float)

for i in range(100): # 任意时刻的概率乘以转移矩阵
    vector1 = vector1.dot(matrix)
    print('Current round: {}'.format(i+1))
    print(np.round(vector1, decimals=3))


for i in range(10): # 转移矩阵自乘结果
    matrix = matrix.dot(matrix)
    print('Current round: {}'.format(i+1))
    print(matrix)