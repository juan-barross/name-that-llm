from pathlib import Path

import pandas as pd

KAGGLE_DIR = Path("/kaggle/input/competitions/name-that-llm")
LOCAL_DIR = Path(__file__).resolve().parents[2] / "data"


def data_dir() -> Path:
    for candidate in (KAGGLE_DIR, LOCAL_DIR):
        if (candidate / "train.csv").exists():
            return candidate
    raise FileNotFoundError(f"train.csv não está em {KAGGLE_DIR} nem em {LOCAL_DIR}")


def load(name: str, root: Path | None = None) -> pd.DataFrame:
    return pd.read_csv((root or data_dir()) / f"{name}.csv", keep_default_na=False, dtype={"text": str})
