from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline

from ntl.features import vectorizer
from ntl.pool import load_pool

SEED = 2027


@dataclass(frozen=True)
class Experiment:
    parts: tuple[str, ...]
    classifier: str = "logreg"
    strength: float = 3.0
    pool_per_label: int = 0
    pool_weight: float = 0.85
    balance: float = 0.0


EXPERIMENTS = {
    "char_logreg": Experiment(("char",)),
    "char_word_logreg": Experiment(("char", "word")),
    "char_word_style_logreg": Experiment(("char", "word", "style")),
    "char15_word_logreg": Experiment(("char15", "word"), strength=10.0),
    "hash_sgd": Experiment(("hash",), "sgd", 2e-5),
    "hash_sgd_pool300": Experiment(("hash",), "sgd", 2e-5, pool_per_label=300),
    "hash_sgd_pool": Experiment(("hash",), "sgd", 2e-5, pool_per_label=2500),
}


def classifier(experiment: Experiment) -> LogisticRegression | SGDClassifier:
    if experiment.classifier == "sgd":
        return SGDClassifier(
            loss="log_loss",
            alpha=experiment.strength,
            class_weight="balanced",
            max_iter=50,
            random_state=SEED,
        )
    return LogisticRegression(C=experiment.strength, class_weight="balanced", max_iter=1000)


def build(name: str) -> Pipeline:
    experiment = EXPERIMENTS[name]
    return Pipeline([("features", vectorizer(*experiment.parts)), ("classifier", classifier(experiment))])


def training_frame(name: str, train: pd.DataFrame) -> pd.DataFrame:
    experiment = EXPERIMENTS[name]
    own = train[["text", "model"]].assign(weight=1.0)
    if not experiment.pool_per_label:
        return own
    pool = load_pool(experiment.pool_per_label).assign(weight=experiment.pool_weight)
    return pd.concat([own, pool], ignore_index=True)


def fit(name: str, train: pd.DataFrame) -> Pipeline:
    data = training_frame(name, train)
    return build(name).fit(data["text"], data["model"], classifier__sample_weight=data["weight"].to_numpy())
