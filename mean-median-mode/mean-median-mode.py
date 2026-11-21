from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    # Write code here

    x = np.asarray(x)
    counts = Counter(x)
    mode = counts.most_common(1)[0][0]
    
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "mode": float(mode)
    }