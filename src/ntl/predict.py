import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ntl.labels import LABELS
from ntl.models import EXPERIMENTS, fit
from ntl.prior import soft_balance

INDEX = {label: i for i, label in enumerate(LABELS)}


def aligned_probabilities(model: Pipeline, texts: pd.Series) -> np.ndarray:
    unknown = set(model.classes_) - set(LABELS)
    if unknown:
        raise ValueError(f"rótulos fora da lista no treino: {sorted(unknown)}")
    probabilities = np.zeros((len(texts), len(LABELS)))
    probabilities[:, [INDEX[label] for label in model.classes_]] = model.predict_proba(texts)
    return probabilities


def decode(probabilities: np.ndarray) -> np.ndarray:
    return np.array(LABELS)[probabilities.argmax(axis=1)]


def fit_predict(experiment: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    probabilities = aligned_probabilities(fit(experiment, train), test["text"])
    return decode(soft_balance(probabilities, EXPERIMENTS[experiment].balance))
