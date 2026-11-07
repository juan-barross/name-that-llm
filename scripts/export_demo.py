import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.pipeline import Pipeline

from ntl.data import load
from ntl.features import FUNCTION_WORDS
from ntl.labels import LABELS
from ntl.models import fit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "modelo"
PARITY = ROOT / "runs" / "paridade.json"
EXPERIMENT = "demo_30k"
PARITY_TEXTS = 300


def view_spec(name: str, step) -> dict:
    terms = sorted(step.vocabulary_, key=step.vocabulary_.get)
    return {
        "nome": name,
        "ngramas": list(step.ngram_range),
        "termos": terms,
        "idf": np.round(step.idf_, 6).tolist(),
    }


def quantize(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales = np.abs(weights).max(axis=1) / 127
    return np.round(weights / scales[:, None]).astype(np.int8), scales


def ordered_weights(model: Pipeline) -> tuple[np.ndarray, np.ndarray]:
    classifier = model.named_steps["classifier"]
    rows = [list(classifier.classes_).index(label) for label in LABELS]
    return classifier.coef_[rows], classifier.intercept_[rows]


def quantized_probabilities(model: Pipeline, texts, codes, scales, intercept) -> np.ndarray:
    features = model.named_steps["features"].transform(texts)
    return softmax(features @ (codes * scales[:, None]).T + intercept, axis=1)


def main() -> None:
    train = load("train")
    model = fit(EXPERIMENT, train)
    weights, intercept = ordered_weights(model)
    codes, scales = quantize(weights)
    views = [view_spec(name, step) for name, step in model.named_steps["features"].transformer_list]
    spec = {
        "rotulos": list(LABELS),
        "palavras_funcionais": sorted(FUNCTION_WORDS),
        "visoes": views,
        "escalas": np.round(scales, 8).tolist(),
        "intercepto": np.round(intercept, 6).tolist(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "modelo.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    (OUT / "pesos.bin").write_bytes(codes.tobytes())

    texts = train["text"].sample(PARITY_TEXTS, random_state=2027).tolist()
    exact = model.predict_proba(texts)[:, [list(model.classes_).index(label) for label in LABELS]]
    quantized = quantized_probabilities(model, texts, codes, scales, intercept)
    PARITY.parent.mkdir(exist_ok=True)
    PARITY.write_text(json.dumps({"textos": texts, "probabilidades": quantized.tolist()}), encoding="utf-8")
    agreement = float((exact.argmax(axis=1) == quantized.argmax(axis=1)).mean())
    size = sum(path.stat().st_size for path in OUT.iterdir()) / 2**20
    summary = {"atributos": int(codes.shape[1]), "mb": round(size, 1), "concordancia_int8": agreement}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
