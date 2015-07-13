import unittest
from models.single_region import get_row_sum, get_col_sum
import numpy


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
