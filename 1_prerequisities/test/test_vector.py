import unittest
import numpy as np
from numpy.testing import assert_array_equal
import sys
sys.path.append('./1_prerequisities/src')
from numpy_array import NumpyArray

class TestVector(unittest.TestCase):
    def setUp(self):
        vector      = [1, 2, 3, 4, 5]
        self.vector = NumpyArray(vector)

    def test_array(self):
        assert_array_equal(np.array([1, 2, 3, 4, 5]), self.vector.array())

if __name__ == '__main__':
    unittest.main()
