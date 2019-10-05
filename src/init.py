import numpy as np


def xavier_uniform(weights, gain=1.0):
    """
    Xavier uniform initializer, it initializes weights and keeps elements of
    weights between [-(gain * sqrt(6 / (node_in + node_out))), gain * sqrt(6 /
    (node_in + node_out))]

    Input:
        weights - a numpy.Array, it is weights should be initialized
        gain - an optional scaling factor
    Output:
        weights - a numpy.Array, it is weights has be initialized
        <So, is it weights has or weights have?>
    """
    node_in, node_out = _get_node_in_and_node_out(weights)
    a = gain * np.sqrt(6.0 / float(node_in + node_out))
    weights = np.random.uniform(-a, a, weights.shape)
    return


def xavier_normal(weights, gain=1.0):
    """
    Xavier initializer, it initializes weights and keeps standard variance of
    the weight is gain * sqrt(2 / (node_in + node_out))

    Input:
        weights - a numpy.Array, it is weights should be initialized
        gain - an optional scaling factor
    Output:
        weights - a numpy.Array, it is weights has be initialized
    """
    node_in, node_out = _get_node_in_and_node_out(weights)
    std = gain * np.sqrt(2.0 / float(node_in + node_out))
    weights = np.random.normal(0.0, std, weights.shape)
    return


# def he_init(node_in, node_out):
#     """
#     He initializer, it initializes weights and keeps variance of
#     the weight is 1 / sqrt(node_in/2)

#     Input:
#         node_in - scalar, shape of initialized weights is (node_in, node_out)
#         node_out - scalar, shape of initialized weights is (node_in, node_out)
#     Output:
#         W - shape (node_in, node_out)
#     """
#     W = np.random.randn(node_in, node_out) / np.sqrt(node_in/2)
#     return W


def _get_node_in_and_node_out(weights):
    """
    """
    dim = len(weights.shape)
    if dim < 2:
        raise ValueError(
            'The dimensions of weights should not be less than 2.')

    node_in = weights.shape[0]
    node_out = weights.shape[1]
    if dim > 2:
        weight_size = np.prod(weights.shape[2:])
        node_in *= weight_size
        node_out *= weight_size
    return node_in, node_out
