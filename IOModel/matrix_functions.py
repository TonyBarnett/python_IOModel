import numpy
from numpy.linalg import inv


def get_row_sum(a: numpy.matrix) -> numpy.matrix:
    return numpy.sum(a, axis=1)


def get_col_sum(a: numpy.matrix) -> numpy.matrix:
    return numpy.sum(a, axis=0)


def element_wise_divide(a: numpy.matrix, b: numpy.matrix) -> numpy.matrix:
    return numpy.divide(numpy.array(a), numpy.array(b))


def matrix_multiply(a: numpy.matrix, b: numpy.matrix) -> numpy.matrix:
    return a.dot(b)


def matrix_divide(a: numpy.matrix, b: numpy.matrix) -> numpy.matrix:
    return matrix_multiply(a, inv(b))


def diagonal(a: numpy.matrix) -> numpy.matrix:
    return numpy.matrix(numpy.diag(a.A1))


def is_matrix_close_to_i(a, tolerance: float) -> bool:
    """
    test whether matrix a is close to the identity matrix
    :param a: a numpy matrix or 2d array, must be square
    :param tolerance:
    :return:
    """
    i = numpy.identity(a.shape[0])

    return (abs(a - i) < tolerance).all()
