import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    
    positional_encoding = []

    for pos in range(seq_len):
        temp = []
        for i in range(d_model):
            if i % 2 == 0:
                temp.append(np.sin(pos/np.pow(base, i / d_model)))
            else:
                temp.append(np.cos(pos/np.pow(base, (i - 1) / d_model)))
        positional_encoding.append(temp)
                            
    return np.asarray(positional_encoding)