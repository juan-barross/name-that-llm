import numpy as np

ITERATIONS = 50


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / matrix.sum(axis=1, keepdims=True)


def sinkhorn(probabilities: np.ndarray, iterations: int = ITERATIONS) -> np.ndarray:
    balanced = np.clip(probabilities, 1e-12, None)
    target = len(balanced) / balanced.shape[1]
    for _ in range(iterations):
        balanced = normalize_rows(balanced * (target / balanced.sum(axis=0)))
    return balanced


def soft_balance(probabilities: np.ndarray, weight: float) -> np.ndarray:
    if not weight:
        return probabilities
    mixed = (
        np.log(np.clip(probabilities, 1e-12, None)) * (1 - weight) + np.log(sinkhorn(probabilities)) * weight
    )
    return normalize_rows(np.exp(mixed - mixed.max(axis=1, keepdims=True)))
