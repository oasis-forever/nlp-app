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

    def test_shape(self):
        self.assertEqual((5,), self.vector.shape())

    def test_slice(self):
        assert_array_equal(np.array([ 1, 2, 3]), self.vector.slice(3))

    def test_sum(self):
        assert_array_equal(
            np.array([2, 4, 6, 8, 10]),
            self.vector.sum(self.vector.array())
        )

    def test_broadcasting_sum(self):
        assert_array_equal(
            np.array([11, 12, 13, 14, 15]),
            self.vector.sum(10)
        )

if __name__ == '__main__':
    unittest.main()
