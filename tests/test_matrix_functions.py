import unittest

import numpy

from IOModel.matrix_functions import get_row_sum, \
    get_col_sum, \
    element_wise_divide, \
    matrix_divide, \
    matrix_multiply, \
    is_matrix_close_to_i


class RowSums(unittest.TestCase):
    def test_type_row_sum(self):
        m = numpy.matrix([1])
        self.assertIs(type(get_row_sum(m)), numpy.matrix)

    def test_empty_matrix_row_sum(self):
        m = numpy.matrix([])
        rs = get_row_sum(m)
        self.assertTrue((rs == []).all())

    def test_single_element_row_sum(self):
        m = numpy.matrix([1])
        rs = get_row_sum(m)
        self.assertTrue((rs == [1]).all())

    def test_vector_row_sum(self):
        m = numpy.matrix([1, 2, 4])
        rs = get_row_sum(m)
        self.assertTrue((rs == [7]).all())

    def test_matrix_row_sum(self):
        m = numpy.matrix([[1, 4], [2, 6]])
        rs = get_row_sum(m)
        self.assertTrue((rs == [[5], [8]]).all())


class ColSums(unittest.TestCase):
    def test_type_col_sum(self):
        m = numpy.matrix([1])
        self.assertIs(type(get_col_sum(m)), numpy.matrix)

    def test_empty_matrix_col_sum(self):
        m = numpy.matrix([])
        rs = get_col_sum(m)
        self.assertTrue((rs == []).all())

    def test_single_element_col_sum(self):
        m = numpy.matrix([1])
        rs = get_col_sum(m)
        self.assertTrue((rs == [1]).all())

    def test_vector_col_sum(self):
        m = numpy.matrix([1, 2, 4])
        rs = get_col_sum(m)
        self.assertTrue((rs == [1, 2, 4]).all())

    def test_matrix_col_sum(self):
        m = numpy.matrix([[1, 4], [2, 6]])
        rs = get_col_sum(m)
        self.assertTrue((rs == [3, 10]).all())


class ElementWiseDivision(unittest.TestCase):
    # def test_type(self):
    #     a = numpy.matrix([2, 4])
    #     b = numpy.matrix([1, 2])
    #     d = element_wise_divide(a, b)
    #     self.assertIs(type(d), numpy.matrix)

    def test_basic(self):
        a = numpy.matrix([3, 4])
        b = numpy.matrix([1, 2])
        d = element_wise_divide(a, b)
        self.assertTrue((d == [3, 2]).all())

    def test_long_arrays(self):
        a = numpy.matrix([3, 4, 10, 100, 9])
        b = numpy.matrix([1, 2, 5, 12, 3])
        d = element_wise_divide(a, b)
        self.assertTrue((d == [3, 2, 2, 25/3, 3]).all())

    def test_matrix(self):
        a = numpy.matrix([[3, 4], [100, 200]])
        b = numpy.matrix([[1, 2], [50, 2]])
        d = element_wise_divide(a, b)
        self.assertTrue((d == [[3, 2], [2, 100]]).all())


class Division(unittest.TestCase):
    def test_type(self):
        a = numpy.matrix([[3.0, 4.0], [10.0, 2.0]])
        b = numpy.matrix([[4.0, 7.0], [2.0, 6.0]])
        d = matrix_divide(a, b)
        self.assertIs(type(d), numpy.matrix)

    def test_simple_float(self):
        a = numpy.matrix([[3.0, 4.0], [10.0, 2.0]])
        b = numpy.matrix([[4.0, 7.0], [2.0, 6.0]])
        # inv(b) = [[0.6, -0.7], [-0.2, 0.4]]

        # [3  4  * [ 0.6 -0.7
        #  10 2]    -0.2  0.4]
        d = matrix_divide(a, b)
        numpy.testing.assert_allclose(d, [[1., -0.5], [5.6, -6.2]])


class Multiply(unittest.TestCase):
    def test_type(self):
        a = numpy.matrix([[3], [4]])
        b = numpy.matrix([1, 2])
        d = matrix_multiply(a, b)
        self.assertIs(type(d), numpy.matrix)

    def test_simple_case(self):
        a = numpy.matrix([1, 2])
        b = numpy.matrix([[3], [4]])
        d = matrix_multiply(a, b)
        numpy.testing.assert_allclose(d, [[11]])

    def test_simple_float(self):
        a = numpy.matrix([1.0, 2.0])
        b = numpy.matrix([[3.0], [4.0]])
        d = matrix_multiply(a, b)
        numpy.testing.assert_allclose(d, [[11.0]])

    def test_multi_dimension(self):
        a = numpy.matrix([[3], [4]])
        b = numpy.matrix([1, 2])
        d = matrix_multiply(a, b)
        numpy.testing.assert_allclose(d, [[3, 6], [4, 8]])

    def test_multi_dimension_float(self):
        a = numpy.matrix([[3.0], [4.0]])
        b = numpy.matrix([1.0, 2.0])
        d = matrix_multiply(a, b)
        numpy.testing.assert_allclose(d, [[3.0, 6.0], [4.0, 8.0]])


class CloseToIdentity(unittest.TestCase):

    def test_type(self):
        is_it = is_matrix_close_to_i(numpy.matrix([[]]), 0)
        self.assertTrue(type(is_it), bool)

    def test_identity(self):
        is_it = is_matrix_close_to_i(numpy.identity(3), 0.0001)
        self.assertTrue(is_it)

    def test_zeros(self):
        is_it = is_matrix_close_to_i(numpy.zeros((3, 3)), 0.1)
        self.assertFalse(is_it)

    def test_ones(self):
        is_it = is_matrix_close_to_i(numpy.ones((3, 3)), 0.1)
        self.assertFalse(is_it)

    def test_near_identity(self):
        is_it = is_matrix_close_to_i(numpy.identity(3) + 0.001 * numpy.identity(3), 0.01)
        self.assertTrue(is_it)
