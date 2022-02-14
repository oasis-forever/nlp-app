import unittest
import numpy as np
from numpy.testing import assert_array_equal
import sys
sys.path.append('./1_prerequisities/src')
from numpy_array import NumpyArray

class TestMatrix(unittest.TestCase):
    def setUp(self):
        matrix = [
            [ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
        self.matrix = NumpyArray(matrix)

    def test_array(self):
        assert_array_equal(
            np.array([
                [ 1,  2,  3,  4],
                [ 5,  6,  7,  8],
                [ 9, 10, 11, 12],
                [13, 14, 15, 16],
            ]),
            self.matrix.array()
        )

    def test_shape(self):
        self.assertEqual((4, 4), self.matrix.shape())

    def test_slice(self):
        assert_array_equal(
            np.array([
                [3, 4],
                [7, 8],
            ]),
            self.matrix.slice(2)
        )

    def test_sum(self):
        assert_array_equal(
            np.array([
                [ 2,  4,  6,  8],
                [10, 12, 14, 16],
                [18, 20, 22, 24],
                [26, 28, 30, 32],
            ]),
            self.matrix.sum(self.matrix.array())
        )

    def test_broadcasting_sum(self):
        assert_array_equal(
            np.array([
                [11, 12, 13, 14],
                [15, 16, 17, 18],
                [19, 20, 21, 22],
                [23, 24, 25, 26],
            ]),
            self.matrix.sum(10)
        )

    def test_substract(self):
        matrix = np.array([
            [ 1, 15, 14,  8],
            [17,  9,  3, 19],
            [16,  8, 19,  8],
            [16,  3,  2, 12],
        ])
        assert_array_equal(
            np.array([
                [  0, -13, -11,  -4],
                [-12,  -3,   4, -11],
                [ -7,   2,  -8,   4],
                [ -3,  11,  13,   4]
            ]),
            self.matrix.substract(matrix)
        )

if __name__ == '__main__':
    unittest.main()
