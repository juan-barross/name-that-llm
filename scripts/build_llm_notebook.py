import argparse
import json
from pathlib import Path

from build_notebook import MODULES, SETUP, cell, module_cell

LLM_MODULES = ["llm_data", "llm"]
INSTALL = "!pip install -q -U peft bitsandbytes"
MAIN = """import json
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from ntl.data import load as load_csv
from ntl.labels import decode
from ntl.llm import LlmRun, load, predict_proba, train
from ntl.llm_data import holdout, training_rows
from ntl.pool import load_pool

SMOKE = False
POOL_PER_LABEL = 1000
POOL_WEIGHT = 0.85
SAIDA = "/kaggle/working"

run = LlmRun(smoke=SMOKE)
fit, valid = holdout(load_csv("train"))
pool = load_pool(POOL_PER_LABEL)
if SMOKE:
    fit, valid = fit.sample(64, random_state=0), valid.sample(32, random_state=0)
    pool = pool.sample(64, random_state=0)
rows = training_rows(fit, valid, pool, POOL_WEIGHT)

inicio = time.perf_counter()
model, tokenizer = load(run)
model = train(run, model, tokenizer, rows)
proba = predict_proba(model, tokenizer, valid["text"].tolist(), run.max_length)
minutos = round((time.perf_counter() - inicio) / 60, 1)

metricas = {
    "balanced_accuracy": round(balanced_accuracy_score(valid["model"], decode(proba)), 4),
    "minutos": minutos,
    "linhas_treino": len(rows),
    "smoke": SMOKE,
}
np.save(f"{SAIDA}/proba_fold0.npy", proba)
with open(f"{SAIDA}/metricas.json", "w") as f:
    json.dump(metricas, f)
model.save_pretrained(f"{SAIDA}/adaptador")
tokenizer.save_pretrained(f"{SAIDA}/adaptador")
metricas"""


def notebook(intro: str) -> dict:
    cells = [
        cell("markdown", intro),
        cell("code", INSTALL),
        cell("code", SETUP),
        *[module_cell(name) for name in [*MODULES, *LLM_MODULES]],
        cell("code", MAIN),
    ]
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Monta o notebook da rodada A no Kaggle a partir de src/ntl")
    parser.add_argument("--saida", type=Path, default=Path("kaggle/llm_rodada_a.ipynb"))
    parser.add_argument("--intro", type=Path, default=Path("kaggle/llm_intro.md"))
    args = parser.parse_args()
    content = notebook(args.intro.read_text(encoding="utf-8"))
    args.saida.write_text(json.dumps(content, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
