import numpy as np


class f_SW:

    def __init__(self, dim=3):

        self.input_dim = dim
        self.bounds = [[-1, 1]] * self.input_dim


    def f(self, x):

        f = 0
        for i in range(len(x)):
            temp = 0
            for j in range(i):
                temp += x[j]
            f += temp ** 2
        return -f


class f_GP:

    def __init__(self, dim=3):

        self.input_dim = dim
        self.bounds = [[1, 4]] * self.input_dim
        self. mu1 = [2] * self.input_dim
        self.mu2 = [3] * self.input_dim
        self.sigma1 = np.eye(self.input_dim)
        self.sigma2 = np.eye(self.input_dim)

    def f(self, x):

        One = ((2 * np.pi) ** (self.input_dim / 2)) * (np.linalg.det(self.sigma1) ** (1/2))

        Two = ((np.array(x) - np.array(self.mu1)).dot(np.linalg.inv(self.sigma1))).dot((np.array(x) - np.array(self.mu1)))

        Three = ((np.array(x) - np.array(self.mu2)).dot(np.linalg.inv(self.sigma2))).dot((np.array(x) - np.array(self.mu2)))
        f = (1 / One) * np.exp((-0.5) * Two) + (1/2) * (1/One) * np.exp((-0.5) * Three)
        return f * 10 **(self.input_dim - 6)



class ackley_func:

    def __init__(self, dim=1000):
        self.input_dim = dim
        self.bounds = [[-32, 32]] * self.input_dim

