import numpy as np

def softmax(x, axis=-1):

    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    batch, query_length, d_model = Q.shape

    assert d_model % num_heads == 0
    d_k = d_model // num_heads

    # 1. Project Q, K, V
    Q_new = np.matmul(Q, W_q)
    K_new = np.matmul(K, W_k)
    V_new = np.matmul(V, W_v)

    # 2. Split into heads
    Q_heads = Q_new.reshape(batch, query_length, num_heads, d_k)
    K_heads = K_new.reshape(batch, query_length, num_heads, d_k)
    V_heads = V_new.reshape(batch, query_length, num_heads, d_k)

    Q_heads = Q_heads.transpose(0, 2, 1, 3)
    K_heads = K_heads.transpose(0, 2, 1, 3)
    V_heads = V_heads.transpose(0, 2, 1, 3)
    
    scores = np.matmul( Q_heads , np.swapaxes(K_heads, -2 , -1))
    scores = scores / np.sqrt(d_k)

    weights = softmax(scores, axis = -1)
    heads = np.matmul(weights, V_heads)

    # 4. Concatenate heads
    heads = heads.transpose(0, 2, 1, 3)
    heads = heads.reshape(batch, query_length, d_model)
    
    output = np.matmul(heads, W_o)

    return output