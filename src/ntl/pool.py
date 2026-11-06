from functools import cache
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ntl.data import LOCAL_DIR

FILE = "ntl_open_pool.parquet"
KAGGLE_INPUT = Path("/kaggle/input")
RISKS = ["flag_gemini37_routing_risk", "flag_near_train_prompt", "flag_train_prompt_regen"]
COLUMNS = ["text", "model", "n_words", "match_weight", *RISKS]
WORDS = (36, 274)
SEED = 2027


def pool_path() -> Path:
    local = LOCAL_DIR / "open_pool" / FILE
    if local.exists():
        return local
    found = sorted(KAGGLE_INPUT.glob(f"**/{FILE}")) if KAGGLE_INPUT.exists() else []
    if not found:
        raise FileNotFoundError(f"{FILE} não está em {local} nem em {KAGGLE_INPUT}")
    return found[0]


def clean(pool: pd.DataFrame) -> pd.DataFrame:
    safe = ~pool[RISKS].any(axis=1)
    sized = pool["n_words"].between(*WORDS)
    return pool[safe & sized & (pool["match_weight"] > 0)]


def sample(pool: pd.DataFrame, per_label: int, seed: int = SEED) -> pd.DataFrame:
    groups = [
        group.sample(n=min(per_label, len(group)), weights=group["match_weight"], random_state=seed)
        for _, group in pool.groupby("model")
    ]
    return pd.concat(groups, ignore_index=True)


@cache
def load_pool(per_label: int) -> pd.DataFrame:
    pool = pq.read_table(pool_path(), columns=COLUMNS).to_pandas()
    return sample(clean(pool), per_label)[["text", "model"]]
