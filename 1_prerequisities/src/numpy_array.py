import numpy as np

class NumpyArray:
    def __init__(self, arr):
        self.np_arr = np.array(arr)

    def array(self):
        return self.np_arr

    def shape(self):
        return self.np_arr.shape

    def sum(self, arr):
        return self.np_arr + arr

    def substract(self, arr):
        return self.np_arr - arr

    def multiply(self, arr):
        return self.np_arr * arr

    def dot_product(self, arr):
        return np.dot(self.np_arr, arr)

    def numpy_sum(self):
        return np.sum(self.np_arr)

