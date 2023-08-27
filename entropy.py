import numpy as np
from matplotlib import pyplot
import math

pyplot.style.use("ggplot")
pyplot.figure(figsize=(10, 5))

# rectified linear function
def H(x):
    return (-1)*math.log2((x**x)*(1-x)**(1-x))

# define a series of inputs
input_series = np.linspace(0,1,1000)
# calculate outputs for our inputs
output_series = [H(x) for x in input_series]

# line plot of raw inputs to rectified outputs
pyplot.plot(input_series, output_series, label="output")
pyplot.legend()
pyplot.title("entropy function")
pyplot.show()
