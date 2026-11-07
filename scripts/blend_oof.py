import argparse
import json
from pathlib import Path

import numpy as np

from ntl.blend import search_weights
from ntl.data import load

RUNS = Path(__file__).resolve().parents[1] / "runs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pesos do blend geométrico sobre as previsões fora da dobra")
    parser.add_argument("experimentos", nargs="*")
    saved = sorted(path.stem.removeprefix("oof_") for path in RUNS.glob("oof_*.npy"))
    names = parser.parse_args().experimentos or saved
    members = [np.load(RUNS / f"oof_{name}.npy") for name in names]
    weights, score = search_weights(members, load("train")["model"].to_numpy())
    chosen = {name: weight for name, weight in zip(names, weights, strict=True) if weight}
    print(json.dumps({"blend": chosen, "cv": round(score, 4)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
