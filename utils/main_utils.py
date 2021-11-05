import numpy as np


def round_array(x, exp):
    """
    Supports exp of every elem and rounds to arbitrary large int values.

    :param x: input array
    :param exp: the exponent used
    :return: the rounded values

    >>> a = round_array([0.28, 0.32], exp=10)
    >>> print(a)
    >>> b = [2, 3]
    >>> np.testing.assert_almost_equal(actual=a, desired=b)
    """
    # return np.array([int(elem * 2 ** exp) for elem in x])
    exp = 0
    shape = x.shape
    x = x.flatten()
    x = np.array([int(elem * 2 ** exp) for elem in x])
    return x.reshape(shape)


def print_array(x):
    """
    Print array x.
    :param x: input array
    :return: print array x
    """
    if len(x.shape) > 1:
        print('[')
        for xi in x:
            print_array(xi)
        print('],')
    else:
        print("[", ",".join([str(elem) for elem in x]), "],")


def array_str(x, out=""):
    """
    Print array x.
    :param x: input array
    :return: print array x
    """
    if len(x.shape) > 1:
        out += '['
        for xi in x:
            out = array_str(x=xi, out=out)
            out += ","
        out = out[:-1]  # remove last comma
        out += ']'
    else:
        out += "["
        out += ",".join([str(elem) for elem in x])
        out += "]"

    return out


if __name__ == "__main__":
    print('1 dim')
    x = np.random.uniform(low=0, high=10, size=(4,))
    print('x: ', x)
    print(array_str(x))

    print('2 dim')
    x = np.random.uniform(low=0, high=10, size=(2, 4))
    print('x: ', x)
    print(array_str(x))

    print('3 dim')
    x = np.random.uniform(low=0, high=10, size=(2, 3, 4))
    print('x: ', x)
    print(array_str(x))