import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

MARKS = {
    "em_dash": "—",
    "en_dash": "–",
    "semicolon": ";",
    "colon": ":",
    "parenthesis": r"\(",
    "curly_quote": "[“”‘’]",
    "straight_quote": "[\"']",
    "ellipsis": r"…|\.\.\.",
    "asterisk": r"\*",
    "hash": "#",
    "digit": r"\d",
    "exclamation": "!",
    "question": r"\?",
    "comma": ",",
    "newline": "\n",
}
PATTERNS = {name: re.compile(pattern) for name, pattern in MARKS.items()}
SENTENCE = re.compile(r"(?<=[.!?])\s+")


def text_style(text: str) -> list[float]:
    words = text.split()
    sentences = [len(s.split()) for s in SENTENCE.split(text) if s.strip()]
    chars = max(len(text), 1)
    rates = [len(pattern.findall(text)) / chars for pattern in PATTERNS.values()]
    return [
        *rates,
        np.log1p(len(words)),
        np.mean([len(w) for w in words]) if words else 0.0,
        np.mean(sentences) if sentences else 0.0,
        np.std(sentences) if sentences else 0.0,
        sum(c.isupper() for c in text) / chars,
    ]


def style_matrix(texts) -> np.ndarray:
    return np.array([text_style(text) for text in texts], dtype=np.float64)


def char_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=False
    )


def word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"\S+"
    )


def style() -> object:
    return make_pipeline(FunctionTransformer(style_matrix), StandardScaler())


def vectorizer(*parts: str) -> FeatureUnion:
    builders = {"char": char_tfidf, "word": word_tfidf, "style": style}
    return FeatureUnion([(part, builders[part]()) for part in parts])
