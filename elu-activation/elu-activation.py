import math

def elu(x: list, alpha: float = 1.0) -> list:
    """
    Returns ELU applied elementwise to the input values.
    """
    # Write code here

    return [
        e if e > 0 else alpha * (math.exp(e) - 1)
        for e in x
    ]
    