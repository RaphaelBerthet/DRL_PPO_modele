from numpy.typing import NDArray
import numpy as np

def softmax(Col: NDArray[np.float32]) -> NDArray[np.float32]:
    Col = Col - np.max(Col, axis=-1, keepdims=True)
    exp_C = np.exp(Col)

    return exp_C / np.sum(exp_C, axis=-1, keepdims=True)
