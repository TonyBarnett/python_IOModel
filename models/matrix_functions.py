import numpy


def get_row_sum(a: numpy.matrix) -> numpy.matrix:
    return numpy.sum(a, axis=1)


def get_col_sum(a: numpy.matrix) -> numpy.matrix:
    return numpy.sum(a, axis=0)

