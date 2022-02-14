import numpy as np

class NumpyArray:
    def __init__(self, arr):
        self.np_arr = np.array(arr)

    def array(self):
        return self.np_arr

    def shape(self):
        return self.np_arr.shape

    def slice(self, idx):
        if len(self.np_arr.shape) == 1:
            return self.np_arr[:idx]
        else:
            return self.np_arr[:idx, idx:]

    def sum(self, arr):
        return self.np_arr + arr

    def substract(self, arr):
        return self.np_arr - arr

