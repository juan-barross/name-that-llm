import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from ntl.data import load
from ntl.predict import decode
from ntl.prior import soft_balance

RUNS = Path(__file__).resolve().parents[1] / "runs"
WEIGHTS = (0.0, 0.15, 0.3, 0.45, 0.6)
DRAWS = 200
SEED = 2027


def shifted_sample(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    classes = np.unique(labels)
    shares = rng.dirichlet(np.full(len(classes), 2.0))
    picks = []
    for label, share in zip(classes, shares, strict=True):
        members = np.flatnonzero(labels == label)
        picks.append(rng.choice(members, size=max(1, round(share * len(members))), replace=False))
    return np.concatenate(picks)


def shifted_scores(probabilities: np.ndarray, labels: np.ndarray) -> dict[float, float]:
    rng = np.random.default_rng(SEED)
    samples = [shifted_sample(labels, rng) for _ in range(DRAWS)]
    return {
        weight: float(
            np.mean(
                [
                    balanced_accuracy_score(labels[s], decode(soft_balance(probabilities[s], weight)))
                    for s in samples
                ]
            )
        )
        for weight in WEIGHTS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Efeito do balanceamento de prior com o teste desbalanceado")
    parser.add_argument("experimento")
    name = parser.parse_args().experimento
    probabilities = np.load(RUNS / f"oof_{name}.npy")
    labels = load("train")["model"].to_numpy()
    for weight, score in shifted_scores(probabilities, labels).items():
        print(f"peso {weight:.2f}: {score:.4f}")


if __name__ == "__main__":
    main()
