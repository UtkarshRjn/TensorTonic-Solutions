import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns the last-axis normalized array.
    """

    mean = np.mean(x, axis = -1, keepdims=True)
    sigma = np.std(x, axis = -1, keepdims=True)

    standard_feature = (x - mean) / (np.sqrt( sigma * sigma + eps))
    
    y = gamma * standard_feature + beta

    return y
    