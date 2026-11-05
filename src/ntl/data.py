from pathlib import Path

import pandas as pd

KAGGLE_DIR = Path("/kaggle/input/competitions/name-that-llm")
LOCAL_DIR = Path("data")


def data_dir() -> Path:
    return KAGGLE_DIR if KAGGLE_DIR.exists() else LOCAL_DIR


def load(name: str, root: Path | None = None) -> pd.DataFrame:
    return pd.read_csv((root or data_dir()) / f"{name}.csv")
