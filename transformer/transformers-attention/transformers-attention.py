import torch
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """

    d_k = Q.shape[-1]
    dot_product_attention = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(d_k)
    scaled_attention = torch.softmax(dot_product_attention, axis=-1)
    attention = torch.matmul(scaled_attention, V)

    return attention