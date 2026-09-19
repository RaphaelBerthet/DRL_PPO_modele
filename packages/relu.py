import numpy as np
from numpy.typing import NDArray

def relu(z: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.maximum(0, z)


def relu_derivative(z: NDArray[np.float32]) -> NDArray[np.float32]:
    return (z > 0).astype(np.float32)