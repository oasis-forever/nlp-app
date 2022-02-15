import numpy as np
from numpy_array import NumpyArray

class Vector(NumpyArray):
    def slice(self, idx):
        return self.np_arr[:idx]
