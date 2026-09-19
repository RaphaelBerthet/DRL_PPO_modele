import numpy as np
from numpy.typing import NDArray

beta1 = np.float32(0.9)
beta2 = np.float32(0.999)
eps = np.float32(1e-8)

def adam_update(param: NDArray[np.float32], grad: NDArray[np.float32], m: NDArray[np.float32], v: NDArray[np.float32], t: int, lr: float):
    m[:] = beta1 * m + (1 - beta1) * grad
    v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param -= lr * m_hat / (np.sqrt(v_hat) + eps)
