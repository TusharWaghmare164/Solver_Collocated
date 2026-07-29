import numpy as np

def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    
    eps = np.finfo(float).eps

    return a / np.where(
        np.abs(b) < eps,
        np.sign(b + eps) * eps,
        b
    )