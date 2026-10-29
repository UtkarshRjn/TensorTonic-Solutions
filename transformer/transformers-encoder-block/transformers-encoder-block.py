import numpy as np

def softmax(x, axis=-1):
    """Provided: numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def ReLU(x):
    return np.maximum(x,0)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns x normalized over the last axis, then scaled by gamma and shifted by beta.
    """

    mean = np.mean(x, axis = -1, keepdims=True)
    sigma = np.std(x, axis = -1, keepdims=True)

    standard_feature = (x - mean) / (np.sqrt( sigma * sigma + eps))
    
    y = gamma * standard_feature + beta

    return y

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns multi-head attention of Q, K, V, projecting them with W_q, W_k, W_v and the merged heads with W_o.
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

    
def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward network output with a ReLU hidden layer.
    """
    
    hidden = np.matmul(x , W1) + b1
    hidden = ReLU(hidden)

    output = np.matmul(hidden, W2) + b2

    return output
    

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """
    
    attention = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    z = layer_norm(x + attention, gamma1, beta1)
    y = layer_norm(z + feed_forward(z, W1, b1, W2, b2) , gamma2, beta2)

    return np.asarray(y, dtype=np.float64)
    
    
    
