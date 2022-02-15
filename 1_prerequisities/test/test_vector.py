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

    def test_substract(self):
        vector = np.array([1, 2, 2, 0, 1])
        assert_array_equal(
            np.array([0, 0, 1, 4, 4]),
            self.vector.substract(vector)
        )

    def test_multiply(self):
        assert_array_equal(
            np.array([1, 4, 9, 16, 25]),
            self.vector.multiply(self.vector.array())
        )

    def test_broadcasting_multiply(self):
        assert_array_equal(
            np.array([2, 4, 6, 8, 10]),
            self.vector.multiply(2)
        )

    def test_dot_product(self):
        vector = np.array([1, 2, 2, 0, 1])
        self.assertEqual(16, self.vector.dot_product(vector))

    def test_max(self):
        self.assertEqual(5, self.vector.max())

    def test_arg_max(self):
        self.assertEqual(4, self.vector.arg_max())

if __name__ == '__main__':
    unittest.main()
