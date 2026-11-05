from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from ntl.labels import LABELS

SEED = 2027
N_SPLITS = 5


@dataclass(frozen=True)
class CrossValidation:
    probabilities: np.ndarray
    fold_scores: list[float]

    @property
    def score(self) -> float:
        return float(np.mean(self.fold_scores))


def folds(groups: pd.Series, n_splits: int = N_SPLITS) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(splitter.split(groups, groups=groups))


def aligned_probabilities(model: Pipeline, texts: pd.Series) -> np.ndarray:
    columns = [list(model.classes_).index(label) for label in LABELS]
    return model.predict_proba(texts)[:, columns]


def cross_validate(build: Callable[[], Pipeline], train: pd.DataFrame) -> CrossValidation:
    probabilities = np.zeros((len(train), len(LABELS)))
    scores = []
    for fit_index, valid_index in folds(train["prompt_id"]):
        model = build().fit(train["text"].iloc[fit_index], train["model"].iloc[fit_index])
        probabilities[valid_index] = aligned_probabilities(model, train["text"].iloc[valid_index])
        predicted = np.array(LABELS)[probabilities[valid_index].argmax(axis=1)]
        scores.append(balanced_accuracy_score(train["model"].iloc[valid_index], predicted))
    return CrossValidation(probabilities, scores)
