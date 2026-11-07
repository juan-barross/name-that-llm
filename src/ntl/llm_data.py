from collections.abc import Iterable

import numpy as np
import pandas as pd

from ntl.labels import LABELS
from ntl.validation import folds

INDEX = {label: i for i, label in enumerate(LABELS)}


def holdout(train: pd.DataFrame, fold: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_index, valid_index = folds(train["prompt_id"])[fold]
    return train.iloc[fit_index], train.iloc[valid_index]


def label_ids(models: Iterable[str]) -> np.ndarray:
    return np.array([INDEX[model] for model in models], dtype=np.int64)


def rows(frame: pd.DataFrame, weight: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"text": frame["text"].to_numpy(), "label": label_ids(frame["model"]), "weight": weight}
    )


def training_rows(
    fit: pd.DataFrame, valid: pd.DataFrame, pool: pd.DataFrame, pool_weight: float
) -> pd.DataFrame:
    clean_pool = pool[~pool["text"].isin(set(valid["text"]))]
    return pd.concat([rows(fit, 1.0), rows(clean_pool, pool_weight)], ignore_index=True)
