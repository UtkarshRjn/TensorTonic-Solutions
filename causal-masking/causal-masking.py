import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    # Write code here
    i = np.arange(scores.shape[-2])[:, None]
    j = np.arange(scores.shape[-1])[None, :]

    mask = np.where(i < j, mask_value, scores)
    
    return mask