import numpy as np
from numpy_array import NumpyArray

class Vector(NumpyArray):
    def slice(self, idx):
        return self.np_arr[:idx]

    def max(self):
        return np.max(self.np_arr)

    def arg_max(self):
        return np.argmax(self.np_arr)

    def random(self, seed, n):
        np.random.seed(seed)
        return np.random.rand(n)

    def initialise_in_zeros(self, n):
        return np.zeros((n,))

    def initialise_in_ones(self, n):
        return np.ones((n,))
