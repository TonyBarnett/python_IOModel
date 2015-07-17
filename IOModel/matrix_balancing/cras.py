from .ras import column_scaling, row_scaling
import numpy
from IOModel.matrix_functions import diagonal, matrix_divide, matrix_multiply, is_matrix_close_to_i


def apply_conditions(a: numpy.matrix, conditions: dict) -> numpy.matrix:
    for cells, value in conditions.items():
        total = value / len(cells)

        for cell in cells:
            a.A1[cell] += total

    return a


def run_cras(row_totals: numpy.matrix,
             column_totals: numpy.matrix,
             conditions: dict,
             maximum_iterations: int=10000,
             tol: float=0.00001) -> numpy.matrix:
    """
    :param row_totals:
    :param column_totals:
    :param conditions: a condition is a group of cells and a total, dict(tuple(tuples): float),
    a cell is (row_index, col_index)
    :param maximum_iterations: big number to stop this running forever
    :param tol: how close is close enough?
    :return:
    """

    a = numpy.matrix(numpy.ones(shape=(len(row_totals), len(column_totals))))
    e = numpy.matrix(numpy.ones(shape=(1, len(row_totals))))
    for _ in range(maximum_iterations):

        r_hat = row_scaling(a, row_totals, e)
        a = matrix_multiply(r_hat, a)

        s_hat = column_scaling(a, column_totals, e)
        a = matrix_multiply(a, s_hat)

        a = apply_conditions(a, conditions)

        if is_matrix_close_to_i(r_hat, tol) and is_matrix_close_to_i(s_hat, tol):
            return a

    return a
