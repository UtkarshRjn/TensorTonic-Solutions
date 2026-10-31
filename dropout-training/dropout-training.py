
import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:

    x = np.asarray(x)

    if rng is None:
        random_values = np.random.random(x.shape)
    else:
        random_values = rng.random(x.shape)

    dropout_pattern = np.where(
        random_values >= p,
        1.0 / (1.0 - p),
        0.0
    )

    output = x * dropout_pattern

    return output, dropout_pattern
