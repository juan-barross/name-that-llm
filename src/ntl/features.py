import re
from collections.abc import Iterable

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
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
STATS = ("log_words", "mean_word_length", "mean_sentence_words", "std_sentence_words", "uppercase_rate")
PATTERNS = {name: re.compile(pattern) for name, pattern in MARKS.items()}
SENTENCE = re.compile(r"(?<=[.!?])\s+")
LETTERS = re.compile(r"[^\W\d_]+")
TOKENS = re.compile(r"\w+|[^\w\s]")
FUNCTION_TEXT = """
    a about above after again against all also although am an and any are as at be because been before
    being below between both but by can could did do does doing down during each either even every few for
    from further had has have having he her here hers him his how however i if in into is it its itself just
    may me might more most must my neither no nor not now of off on once only or other our ours out over own
    perhaps rather same shall she should so some such than that the their theirs them then there therefore
    these they this those though through thus to too under until up upon us very was we were what when where
    whether which while who whom whose why will with within without would yet you your yours
"""
FUNCTION_WORDS = frozenset(FUNCTION_TEXT.split())


def word_shape(text: str) -> str:
    return LETTERS.sub(lambda match: "X" if match[0][0].isupper() else "x", text)


def skeleton_token(token: str) -> str:
    word = token.lower()
    return word if word in FUNCTION_WORDS or not token.isalnum() else "CONTENT"


def function_skeleton(text: str) -> str:
    return " ".join(skeleton_token(token) for token in TOKENS.findall(text))


def text_style(text: str) -> list[float]:
    words = text.split()
    sentences = [len(s.split()) for s in SENTENCE.split(text) if s.strip()] or [0]
    chars = max(len(text), 1)
    rates = [len(pattern.findall(text)) / chars for pattern in PATTERNS.values()]
    stats = [
        np.log1p(len(words)),
        np.mean([len(w) for w in words]) if words else 0.0,
        np.mean(sentences),
        np.std(sentences),
        sum(c.isupper() for c in text) / chars,
    ]
    return rates + stats


def style_matrix(texts: Iterable[str]) -> np.ndarray:
    return np.array([text_style(text) for text in texts], dtype=np.float32)


def char_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, lowercase=False, dtype=np.float32
    )


def word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        token_pattern=r"\S+",
        dtype=np.float32,
    )


def char15_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char", ngram_range=(1, 5), min_df=3, sublinear_tf=True, lowercase=False, dtype=np.float32
    )


def char_hashing() -> Pipeline:
    hashing = HashingVectorizer(
        analyzer="char",
        ngram_range=(1, 5),
        n_features=2**20,
        alternate_sign=False,
        norm=None,
        lowercase=False,
    )
    return make_pipeline(hashing, TfidfTransformer(sublinear_tf=True))


def shape_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=word_shape,
        analyzer="char",
        ngram_range=(2, 6),
        min_df=2,
        sublinear_tf=True,
        lowercase=False,
        dtype=np.float32,
    )


def function_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=function_skeleton,
        analyzer="word",
        ngram_range=(1, 3),
        min_df=2,
        sublinear_tf=True,
        lowercase=False,
        token_pattern=r"\S+",
        dtype=np.float32,
    )


def scaled_style() -> Pipeline:
    return make_pipeline(FunctionTransformer(style_matrix), StandardScaler())


BUILDERS = {
    "char": char_tfidf,
    "char15": char15_tfidf,
    "hash": char_hashing,
    "word": word_tfidf,
    "shape": shape_tfidf,
    "function": function_tfidf,
    "style": scaled_style,
}


def vectorizer(*parts: str) -> FeatureUnion:
    return FeatureUnion([(part, BUILDERS[part]()) for part in parts])
