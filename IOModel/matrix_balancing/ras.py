import numpy
from IOModel.matrix_functions import diagonal, matrix_divide, matrix_multiply, is_matrix_close_to_i


def column_scaling(a: numpy.matrix,
                   column_totals: numpy.matrix,
                   e: numpy.matrix) -> numpy.matrix:

    s_hat = matrix_divide(diagonal(column_totals),
                          diagonal(matrix_multiply(e.T, a)))

    return s_hat


def row_scaling(a: numpy.matrix,
                row_totals: numpy.matrix,
                e: numpy.matrix) -> numpy.matrix:
    r_hat = matrix_divide(diagonal(row_totals),
                          diagonal(matrix_multiply(a, e)))

    return r_hat


def run_ras(row_totals: numpy.matrix,
            column_totals: numpy.matrix,
            maximum_iterations: int=10000,
            a: numpy.matrix=None,
            tol: float=0.00001) -> numpy.matrix:
    """
    creates a matrix who's respective rows sum to the row_totals and who's respective columns sum to the column_totals
    within the tolerance.
    :param row_totals:
    :param column_totals:
    :param maximum_iterations: big number to stop this running forever
    :param tol: how close is close enough?
    :return:
    """
    if not a:
        a = numpy.ones(shape=(len(row_totals), len(column_totals)))
    e = numpy.ones(shape=(1, len(row_totals)))

    for _ in range(maximum_iterations):

        r_hat = row_scaling(a, row_totals, e)
        a = matrix_multiply(r_hat, a)

        s_hat = column_scaling(a, column_totals, e)
        a = matrix_multiply(a, s_hat)

        if is_matrix_close_to_i(r_hat, tol) and is_matrix_close_to_i(s_hat, tol):
            return a

    return a
