import numpy as np

class NumpyArray:
    def __init__(self, arr):
        self.np_arr = np.array(arr)

    def array(self):
        return self.np_arr

    def shape(self):
        return self.np_arr.shape

