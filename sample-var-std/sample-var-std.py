import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here

    x = np.asarray(x)
    x = x - np.mean(x)

    n = x.shape[0]
    
    variance = np.sum(x * x) / (n-1)
    std = np.sqrt(variance)
    
    return {
        "variance": float(variance),
        "standard_deviation": float(std)
    }

    