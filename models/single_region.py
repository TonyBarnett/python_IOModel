import numpy
#  TODO get UK Consumption, UK production, UK emissions and return some carbon intensities
#  q is column sum of V
#  g is row sum of V
#  y is row sum of U


def get_row_sum(a: numpy.matrix) -> numpy.matrix:
    raise NotImplementedError()


def get_col_sum(a: numpy.matrix) -> numpy.matrix:
    return get_row_sum(a.transpose())


def run_model(consumption, production, emissions) -> numpy.matrix:
    """
    :param consumption: U
    :param production: V
    :param emissions: e
    :return:
    """
    numpy.matrix.diagonal()
