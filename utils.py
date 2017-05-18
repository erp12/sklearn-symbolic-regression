"""
"""

import math, random

MAX_NUM_MAGNITUDE = 1e8
MIN_NUM_MAGNITUDE = 1e-8

def get_arity(f):
    """Returns the artity of the function f.

    :returns: Number of argumetns f takes.
    :example:
        >>> get_arity(lambda x: x*x)
        1
        >>> get_arity(lambda x,y: x*y)
        2
    """
    return f.__code__.co_argcount

def noise_factor():
    """Returns Gaussian noise of mean 0, std dev 1.

    :returns: Float samples from Gaussian distribution.
    :example:
        >>> noise_factor()
        1.43412557975
        >>> noise_factor()
        -0.0410900866765
    """
    return math.sqrt(-2.0 * math.log(random.random())) * math.cos(2.0 * math.pi * random.random())

def keep_number_reasonable(n):
    """TODO: write docstring
    """
    if n > MAX_NUM_MAGNITUDE:
        return float(MAX_NUM_MAGNITUDE)
    elif n < -MAX_NUM_MAGNITUDE:
        return float(-MAX_NUM_MAGNITUDE)
    elif n < MIN_NUM_MAGNITUDE and n > -MIN_NUM_MAGNITUDE:
        return 0
    else:
        return n
