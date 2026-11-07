from collections.abc import Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from ntl.labels import decode
from ntl.prior import normalize_rows

GRID = tuple(np.round(np.linspace(0.1, 1.0, 10), 2))
MIN_GAIN = 0.005


def geometric_blend(members: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    logits = sum(w * np.log(np.clip(p, 1e-12, None)) for p, w in zip(members, weights, strict=True))
    return normalize_rows(np.exp(logits - logits.max(axis=1, keepdims=True)))


def blend_score(members: Sequence[np.ndarray], weights: Sequence[float], labels: np.ndarray) -> float:
    return balanced_accuracy_score(labels, decode(geometric_blend(members, weights)))


def search_weights(members: Sequence[np.ndarray], labels: np.ndarray) -> tuple[list[float], float]:
    singles = [blend_score([m], [1.0], labels) for m in members]
    weights = [0.0] * len(members)
    weights[int(np.argmax(singles))] = 1.0
    best = max(singles)
    improved = True
    while improved:
        improved = False
        for i in range(len(members)):
            for w in GRID:
                trial = [*weights[:i], w, *weights[i + 1 :]]
                score = blend_score(members, trial, labels)
                if score > best + MIN_GAIN:
                    weights, best, improved = trial, score, True
    return weights, best
