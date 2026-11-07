import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_notebook", ROOT / "scripts" / "build_notebook.py")
build_notebook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_notebook)


LLM_MODULES = {"llm", "llm_data"}


def test_notebook_leva_todos_os_modulos_do_pacote():
    package = {path.stem for path in (ROOT / "src" / "ntl").glob("*.py")}

    assert {*build_notebook.MODULES, "__init__"} == package - LLM_MODULES


def test_nenhuma_celula_writefile_fica_vazia(monkeypatch):
    monkeypatch.chdir(ROOT)
    cells = build_notebook.notebook({"char_logreg": 1.0}, 0.3, "# intro")["cells"]

    for cell in cells:
        source = "".join(cell["source"])
        if source.startswith("%%writefile"):
            assert source.split("\n", 1)[1].strip()


def test_codigo_do_notebook_compila(monkeypatch):
    monkeypatch.chdir(ROOT)
    cells = build_notebook.notebook({"char_logreg": 1.0, "hash_sgd": 0.4}, 0.3, "# intro")["cells"]

    for cell in cells[1:]:
        source = "".join(cell["source"])
        body = source.split("\n", 1)[1] if source.startswith("%%writefile") else source
        compile(body, "celula", "exec")
    main = "".join(cells[-1]["source"])
    assert "BLEND = {'char_logreg': 1.0, 'hash_sgd': 0.4}" in main
    assert "BALANCE = 0.3" in main


def test_membro_sem_peso_vale_um():
    assert build_notebook.parse_member("hash_sgd_pool") == ("hash_sgd_pool", 1.0)
    assert build_notebook.parse_member("char_logreg=0.4") == ("char_logreg", 0.4)
