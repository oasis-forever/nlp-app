import numpy as np
from numpy_array import NumpyArray

class Matrix(NumpyArray):
    def slice(self, idx1, idx2):
        return self.np_arr[:idx1, idx2:]

    def max(self, axis):
        return np.max(self.np_arr, axis=axis)

    def arg_max(self, axis):
        return np.argmax(self.np_arr, axis=axis)
