import numpy as np

def factor(number):
    """Function that calculates factors of a number.

    Parameters
    ----------
    number : int
        Number to calculate factors of.

    Returns
    -------
    ndarray
        Array with factors of a number.
    """
    products = []
    product = 2
    while product*product <= number:
        while number % product == 0:
            products.append(product)
            number = number//product
        product += 1    
    return np.array(products)[::-1]