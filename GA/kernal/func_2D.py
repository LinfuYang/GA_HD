import numpy as np


class f_1:

    def __init__(self, dim=2):

        self.input_dim = dim
        self.bounds = [[-3.0, 12.1], [4.1, 5.8]]


    def f(self, x):

        f = 21.5 + x[0] * np.sin(4 * np.pi * x[0]) + x[1] * np.sin(20 * np.pi * x[1])
        return f
