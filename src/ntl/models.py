from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ntl.features import vectorizer

EXPERIMENTS = {
    "char_logreg": (("char",), 10.0),
    "char_word_logreg": (("char", "word"), 10.0),
    "char_word_style_logreg": (("char", "word", "style"), 10.0),
}


def build(name: str) -> Pipeline:
    parts, strength = EXPERIMENTS[name]
    classifier = LogisticRegression(C=strength, class_weight="balanced", max_iter=3000)
    return Pipeline([("features", vectorizer(*parts)), ("classifier", classifier)])
