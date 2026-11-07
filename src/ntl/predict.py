import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ntl.blend import geometric_blend
from ntl.labels import LABELS, decode
from ntl.models import fit
from ntl.prior import soft_balance

INDEX = {label: i for i, label in enumerate(LABELS)}


def aligned_probabilities(model: Pipeline, texts: pd.Series) -> np.ndarray:
    unknown = set(model.classes_) - set(LABELS)
    if unknown:
        raise ValueError(f"rótulos fora da lista no treino: {sorted(unknown)}")
    probabilities = np.zeros((len(texts), len(LABELS)))
    probabilities[:, [INDEX[label] for label in model.classes_]] = model.predict_proba(texts)
    return probabilities


def fit_predict(
    blend: dict[str, float], train: pd.DataFrame, test: pd.DataFrame, balance: float
) -> np.ndarray:
    members = [aligned_probabilities(fit(name, train), test["text"]) for name in blend]
    return decode(soft_balance(geometric_blend(members, list(blend.values())), balance))
