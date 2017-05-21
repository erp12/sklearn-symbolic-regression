"""
"""

import math, random
import numpy as np

MAX_NUM_MAGNITUDE = 1e8
MIN_NUM_MAGNITUDE = 1e-8

def get_arity(f):
    """Returns the artity of the function f.

    Returns
    -------
    arity : int
        Returns the number of argumetns f takes.

    Examples
    --------
    ::
        >>> get_arity(lambda x: x*x)
        1
        >>> get_arity(lambda x,y: x*y)
        2
    """
    return f.__code__.co_argcount

def noise_factor():
    """Returns Gaussian noise of mean 0, std dev 1.

    Returns
    -------
    n : float
        Returns a sample from Gaussian distribution.

    Examples
    --------
    ::
        >>> noise_factor()
        1.43412557975
        >>> noise_factor()
        -0.0410900866765
    """
    a = math.sqrt(-2.0 * math.log(random.random()))
    b = math.cos(2.0 * math.pi * random.random())
    return a * b

def keep_number_reasonable(n):
    """Clamps n to be a valid number. This is used to stop evolution from
    producing overflow errors.

    Parameters
    ----------
    n : {int, float}
        Number to clamp.

    Returns
    -------
    n : float
        Returns n clamped to a resonable range.

    Warnings
    --------
    Note that this function currently relies on global variables, which is
    probably not ideal. A better way should be found.
    """
    if n > MAX_NUM_MAGNITUDE:
        return float(MAX_NUM_MAGNITUDE)
    elif n < -MAX_NUM_MAGNITUDE:
        return float(-MAX_NUM_MAGNITUDE)
    elif n < MIN_NUM_MAGNITUDE and n > -MIN_NUM_MAGNITUDE:
        return 0
    else:
        return n

def median_absolute_deviation(a):
    """Returns the MAD of X.

    Parameters
    ----------
    a : array-like, shape = (n,)

    Returns
    -------
    mad : float
    """
    return np.median(np.abs(a - np.median(a)))
