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
    "blend",
    "validation",
    "submission",
]
MAIN = """from ntl.data import load
from ntl.predict import fit_predict
from ntl.submission import write

BLEND = {blend}
BALANCE = {balance}

train, test = load("train"), load("test")
submission = write(test["id"], fit_predict(BLEND, train, test, BALANCE))
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


def parse_member(text: str) -> tuple[str, float]:
    name, _, weight = text.partition("=")
    return name, float(weight or 1.0)


def notebook(blend: dict[str, float], balance: float, intro: str) -> dict:
    main = MAIN.replace("{blend}", repr(blend)).replace("{balance}", repr(balance))
    cells = [
        cell("markdown", intro),
        cell("code", 'from pathlib import Path\n\nPath("ntl").mkdir(exist_ok=True)'),
        *[module_cell(name) for name in MODULES],
        cell("code", main),
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
    parser.add_argument("membros", nargs="+", type=parse_member, help="experimento=peso")
    parser.add_argument("--balanco", type=float, default=0.3)
    parser.add_argument("--saida", type=Path, default=Path("kaggle/submissao.ipynb"))
    parser.add_argument("--intro", type=Path, default=Path("kaggle/intro.md"))
    args = parser.parse_args()
    content = notebook(dict(args.membros), args.balanco, args.intro.read_text(encoding="utf-8"))
    args.saida.write_text(json.dumps(content, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
