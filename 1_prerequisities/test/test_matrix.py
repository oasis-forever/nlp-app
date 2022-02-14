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

if __name__ == '__main__':
    unittest.main()
