import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ntl.data import load
from ntl.models import EXPERIMENTS, build
from ntl.validation import cross_validate

RUNS = Path(__file__).resolve().parents[1] / "runs"


def run(name: str, train: pd.DataFrame) -> dict[str, float | str]:
    start = time.perf_counter()
    result = cross_validate(lambda: build(name), train)
    np.save(RUNS / f"oof_{name}.npy", result.probabilities)
    return {
        "experimento": name,
        "cv": round(result.score, 4),
        "desvio": round(float(np.std(result.fold_scores)), 4),
        "minutos": round((time.perf_counter() - start) / 60, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validação cruzada por prompt dos experimentos")
    parser.add_argument("experimentos", nargs="*", default=list(EXPERIMENTS))
    names = parser.parse_args().experimentos
    RUNS.mkdir(exist_ok=True)
    train = load("train")
    with (RUNS / "experimentos.jsonl").open("a", encoding="utf-8") as log:
        for name in names:
            line = run(name, train)
            log.write(json.dumps(line, ensure_ascii=False) + "\n")
            print(line)


if __name__ == "__main__":
    main()
