def compute_discounted_sum(x, gamma):
    """
    Compute discounted sum of future values
    out = x[0] + gamma * x[1] + gamma^2 * x[2] + ...

    :param x: input array/list
    :param gamma: decay rate
    :return an output to apply the discount rates
    """
    _output = 0.0
    for i in reversed(x):
        _output *= gamma
        _output += i
    return _output
