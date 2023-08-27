from matplotlib import pyplot

pyplot.style.use("ggplot")
pyplot.figure(figsize=(10, 5))

# rectified linear function
def Leaky_ReLU(x):
    if x > 0:
        return x
    else:
        return 0.01 * x


def ReLU(x):
    if x > 0:
        return x
    else:
        return 0


def SmoothReLU(x):
    if x < -1:
        return 0.5 * x
    elif x >= -1 and x <= 1:
        return (x**2) / 4 + 0.5
    elif x > 1:
        return x - 0.5


# define a series of inputs
input_series = [x for x in range(-19, 19)]
# calculate outputs for our inputs
leaky_output_series = [Leaky_ReLU(x) for x in input_series]
ordinary_output_series = [ReLU(x) for x in input_series]
smooth_output_series = [SmoothReLU(x) for x in input_series ]
# line plot of raw inputs to rectified outputs
pyplot.plot(input_series, ordinary_output_series, label="ordinary output")
pyplot.plot(input_series, leaky_output_series, label="leaky output")
pyplot.plot(input_series,smooth_output_series,label='smooth output')
pyplot.legend()
pyplot.title("ReLU and its variants")
pyplot.show()
