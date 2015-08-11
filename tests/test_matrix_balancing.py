import unittest
import numpy
from IOModel.matrix_balancing.ras import row_scaling, column_scaling, run_ras
from IOModel.matrix_balancing.cras import apply_conditions
from IOModel.matrix_functions import get_row_sum, get_col_sum


class RowScaling(unittest.TestCase):
    @staticmethod
    def _get_row_totals() -> numpy.matrix:
        return numpy.matrix([[10], [20]])

    @staticmethod
    def _get_matrix_to_balance() -> numpy.matrix:
        return numpy.matrix([[1, 1], [1, 1]])

    @staticmethod
    def _get_column_ones() -> numpy.matrix:
        return numpy.matrix(numpy.ones((2, 1)))

    def setUp(self):
        self.totals = RowScaling._get_row_totals()
        self.m = RowScaling._get_matrix_to_balance()
        self.e = RowScaling._get_column_ones()

    def test_type(self):
        result = row_scaling(self.m, self.totals, self.e)
        self.assertIs(type(result), numpy.matrix)

    def test_simple_case(self):
        r_hat = row_scaling(self.m, self.totals, self.e)

        self.assertTrue((r_hat == [[5, 0], [0, 10]]).all())


class ColumnScaling(unittest.TestCase):
    @staticmethod
    def _get_row_totals() -> numpy.matrix:
        return numpy.matrix([[10, 20]])

    @staticmethod
    def _get_matrix_to_balance() -> numpy.matrix:
        return numpy.matrix([[1, 1], [1, 1]])

    @staticmethod
    def _get_column_ones() -> numpy.matrix:
        return numpy.matrix(numpy.ones((2, 1)))

    def setUp(self):
        totals = ColumnScaling._get_row_totals()
        m = ColumnScaling._get_matrix_to_balance()
        e = ColumnScaling._get_column_ones()
        return totals, m, e

    def test_type(self):
        totals, m, e = self.setUp()
        result = column_scaling(m, totals, e)

        self.assertIs(type(result), numpy.matrix)

    def test_simple_case(self):
        totals, m, e = self.setUp()
        r_hat = column_scaling(m, totals, e)

        self.assertTrue((r_hat == [[5, 0], [0, 10]]).all())


class RAS(unittest.TestCase):
    def setUp(self):
        self.matrix = numpy.matrix([[1, 2], [3, 4]])
        self.row_sums = get_row_sum(self.matrix)
        self.column_sums = get_col_sum(self.matrix)

    def test_simple_case(self):
        output = run_ras(self.row_sums, self.column_sums)
        r = get_row_sum(output)
        c = get_col_sum(output)

        self.assertTrue((get_row_sum(output) == self.row_sums).all())
        self.assertTrue((get_col_sum(output) == self.column_sums).all())


class CRAS(unittest.TestCase):
    def setUp(self):
        self.matrix = numpy.matrix([[1, 2, 3],
                                    [5, 7, 9],
                                    [3, 2, 5]])

        self.conditions = {(0, 0): 4, (0, 1): 5}

    def test_simple_case(self):
        m = apply_conditions(self.matrix, self.conditions)
        self.assertEqual(m[(0, 0)], 4)
        self.assertEqual(m[(0, 1)], 5)
