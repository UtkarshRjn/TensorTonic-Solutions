import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """

    positional_encoding = []

    for pos in range(seq_length):
        temp = []
        for i in range(d_model):
            if i % 2 == 0:
                temp.append(np.sin(pos/np.pow(10000, i / d_model)))
            else:
                temp.append(np.cos(pos/np.pow(10000, (i - 1) / d_model)))
        positional_encoding.append(temp)
                            
    return np.asarray(positional_encoding)