import numpy
from numpy.linalg import inv


def get_row_sum(a: numpy.matrix) -> numpy.matrix:
    return numpy.sum(a, axis=1)


def get_col_sum(a: numpy.matrix) -> numpy.matrix:
    return numpy.sum(a, axis=0)


def element_wise_divide(a: numpy.matrix, b: numpy.matrix) -> numpy.matrix:
    return numpy.divide(a, b)


def matrix_multiply(a: numpy.matrix, b: numpy.matrix) -> numpy.matrix:
    return a.dot(b)


def matrix_divide(a: numpy.matrix, b: numpy.matrix) -> numpy.matrix:
    return matrix_multiply(a, inv(b))
