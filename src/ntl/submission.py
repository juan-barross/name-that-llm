from pathlib import Path

import pandas as pd

from ntl.labels import LABELS

COLUMNS = ["id", "model"]


def validate(submission: pd.DataFrame) -> None:
    if list(submission.columns) != COLUMNS:
        raise ValueError(f"colunas devem ser {COLUMNS}, vieram {list(submission.columns)}")
    if submission.isna().any().any():
        raise ValueError("submissão com valor nulo")
    if submission["id"].duplicated().any():
        raise ValueError("submissão com id repetido")
    unknown = set(submission["model"]) - set(LABELS)
    if unknown:
        raise ValueError(f"rótulos fora da lista: {sorted(unknown)}")


def write(ids: pd.Series, predictions, path: Path = Path("submission.parquet")) -> pd.DataFrame:
    submission = pd.DataFrame({"id": ids.to_numpy(), "model": list(predictions)})
    validate(submission)
    submission.to_parquet(path, index=False)
    return submission
