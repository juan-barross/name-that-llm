import numpy as np
import pandas as pd
import pytest

from ntl.labels import LABELS

TOPICS = ["entropia", "evolução", "recursão", "ética", "galáxias", "poesia"]


def passage(label_index: int, topic: str, rng: np.random.Generator) -> str:
    tic = f"marca{label_index}"
    filler = " ".join(rng.choice(["isso", "aquilo", "também", "porém", "logo"], size=12))
    return f"Sobre {topic}, {tic} {filler} {tic}."


@pytest.fixture
def train() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = [
        {
            "id": f"{p}-{k}",
            "prompt_id": f"p{p}",
            "text": passage(k, TOPICS[p % len(TOPICS)], rng),
            "model": label,
        }
        for p in range(10)
        for k, label in enumerate(LABELS)
    ]
    return pd.DataFrame(rows)
