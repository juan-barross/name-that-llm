from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from ntl.labels import LABELS
from ntl.models import fit
from ntl.predict import aligned_probabilities, decode

SEED = 2027
N_SPLITS = 5


@dataclass(frozen=True)
class CrossValidation:
    probabilities: np.ndarray
    fold_scores: tuple[float, ...]

    @property
    def score(self) -> float:
        return float(np.mean(self.fold_scores))


def folds(groups: pd.Series, n_splits: int = N_SPLITS) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(splitter.split(groups, groups=groups))


def cross_validate(experiment: str, train: pd.DataFrame) -> CrossValidation:
    probabilities = np.zeros((len(train), len(LABELS)))
    scores = []
    for fit_index, valid_index in folds(train["prompt_id"]):
        valid = train.iloc[valid_index]
        model = fit(experiment, train.iloc[fit_index])
        probabilities[valid_index] = aligned_probabilities(model, valid["text"])
        scores.append(balanced_accuracy_score(valid["model"], decode(probabilities[valid_index])))
    return CrossValidation(probabilities, tuple(scores))
