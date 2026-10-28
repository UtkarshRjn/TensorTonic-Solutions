import math

def selu(x: list) -> list:
    """
    Returns SELU values rounded to four decimal places.
    """
    # Write code here
    lmbd = 1.0507
    alpha = 1.6733

    return [
        lmbd * e if e > 0
        else lmbd * alpha * (math.exp(e) - 1)
        for e in x
    ]
    