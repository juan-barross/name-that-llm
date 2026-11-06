import argparse
import json
import uuid
from pathlib import Path

PACKAGE = Path("src/ntl")
MODULES = [
    "__init__",
    "labels",
    "data",
    "pool",
    "prior",
    "features",
    "models",
    "predict",
    "validation",
    "submission",
]
MAIN = """from ntl.data import load
from ntl.predict import fit_predict
from ntl.submission import write

EXPERIMENT = "{experiment}"

train, test = load("train"), load("test")
submission = write(test["id"], fit_predict(EXPERIMENT, train, test))
submission["model"].value_counts()"""


def cell(kind: str, source: str) -> dict:
    base = {
        "cell_type": kind,
        "id": uuid.uuid4().hex,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    return base | {"execution_count": None, "outputs": []} if kind == "code" else base


def module_cell(name: str) -> dict:
    return cell("code", f"%%writefile ntl/{name}.py\n" + (PACKAGE / f"{name}.py").read_text(encoding="utf-8"))


def notebook(experiment: str, intro: str) -> dict:
    cells = [
        cell("markdown", intro),
        cell("code", 'from pathlib import Path\n\nPath("ntl").mkdir(exist_ok=True)'),
        *[module_cell(name) for name in MODULES],
        cell("code", MAIN.replace("{experiment}", experiment)),
    ]
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monta o notebook de submissão do Kaggle a partir de src/ntl"
    )
    parser.add_argument("experimento")
    parser.add_argument("--saida", type=Path, default=Path("kaggle/submissao.ipynb"))
    parser.add_argument("--intro", type=Path, default=Path("kaggle/intro.md"))
    args = parser.parse_args()
    content = notebook(args.experimento, args.intro.read_text(encoding="utf-8"))
    args.saida.write_text(json.dumps(content, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
