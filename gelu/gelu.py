import math
import numpy as np
from scipy.special import erf

def gelu(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here

    x = np.array(x)

    return x * (1 + erf(x/math.sqrt(2)))/ 2 