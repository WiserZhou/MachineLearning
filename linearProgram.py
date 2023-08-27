# coding=utf-8
import math
from scipy.optimize import minimize
import numpy as np


def f(x):
    return x * math.log(x, 2)


# demo 2
# 计算  (2+x1)/(1+x2) - 3*x1+4*x3 的最小值  x1,x2,x3的范围都在0.1到0.9 之间
def fun():
    v = lambda x: sum([f(xi) for xi in x[:32]])
    return v


def con(args):
    # 约束条件 分为eq 和ineq
    # eq表示 函数结果等于0 ； ineq 表示表达式大于等于0
    cons = []

    for i in range(32):
        x_min, x_max = args[2 * i], args[2 * i + 1]
        cons.append({"type": "ineq", "fun": lambda x, i=i: x[i] - x_min})
        cons.append({"type": "ineq", "fun": lambda x, i=i: -x[i] + x_max})
        # 32个变量之和为1约束条件
    cons.append({"type": "eq", "fun": lambda x: sum(x) - 1})
    return cons


MIN = 0.00001
MAX = 1 - MIN

if __name__ == "__main__":

    args2 = [MIN, MAX]
    args3 = []
    for i in range(32):
        args3.extend(args2)
    args3 = tuple(args3)
    # 设置参数范围/约束条件
    args1 = args3  # x1min, x1max, x2min, x2max
    cons = con(args1)
    # 设置初始猜测值
    x0 = np.asarray(
        (
            1 / 32 - 0.01,
            1 / 32 + 0.01,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
            1 / 32,
        )
    )

    res = minimize(fun(), x0, method="SLSQP", constraints=cons)
    print(res.fun)
    print(res.success)
    print(res.x)
    print(res)
