import numpy as np
import pandas as pd

from ntl.labels import LABELS
from ntl.models import build
from ntl.validation import aligned_probabilities


def fit_predict(experiment: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = build(experiment).fit(train["text"], train["model"])
    return np.array(LABELS)[aligned_probabilities(model, test["text"]).argmax(axis=1)]
