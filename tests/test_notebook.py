import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_notebook", ROOT / "scripts" / "build_notebook.py")
build_notebook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_notebook)


def test_notebook_leva_todos_os_modulos_do_pacote():
    package = {path.stem for path in (ROOT / "src" / "ntl").glob("*.py")}

    assert set(build_notebook.MODULES) == package


def test_codigo_do_notebook_compila(monkeypatch):
    monkeypatch.chdir(ROOT)
    cells = build_notebook.notebook("char_word_logreg", "# intro")["cells"]

    for cell in cells[1:]:
        source = "".join(cell["source"])
        body = source.split("\n", 1)[1] if source.startswith("%%writefile") else source
        compile(body, "celula", "exec")
    assert 'EXPERIMENT = "char_word_logreg"' in "".join(cells[-1]["source"])
