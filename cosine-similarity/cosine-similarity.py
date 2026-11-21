import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here

    a = np.asarray(a)
    b = np.asarray(b)

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))