import numpy
from numpy.linalg import inv

from IOModel.matrix_functions import get_col_sum, get_row_sum, matrix_divide, matrix_multiply, element_wise_divide, diagonal


#  TODO get UK Consumption, UK production, UK emissions and return some carbon intensities
#  q is column sum of V
#  g is row sum of V
#  y is row sum of U


def run_model(consumption: numpy.matrix, production: numpy.matrix, emissions: numpy.matrix) -> numpy.matrix:
    """
    :param consumption: U
    :param production: V
    :param emissions: e
    :return:
    """
    i = numpy.identity(len(consumption))

    q = get_row_sum(production.source_data.elements)
    g = get_col_sum(production.source_data.elements)

    y = get_row_sum(consumption.source_data.elements)

    b = matrix_divide(consumption.source_data.elements, diagonal(g))
    d = matrix_divide(production.source_data.elements, diagonal(q))

    a = matrix_multiply(b, d)

    money_part = matrix_multiply(inv(i - a), y)

    return element_wise_divide(emissions.source_data.elements, money_part)
