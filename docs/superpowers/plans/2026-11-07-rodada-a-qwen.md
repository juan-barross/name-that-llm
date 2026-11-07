# Rodada A: Qwen3-4B com LoRA como classificador — Plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Um notebook do Kaggle (GPU T4) que ajusta o Qwen3-4B com LoRA como classificador dos 18 rótulos, mede a balanced accuracy numa dobra por prompt e salva as probabilidades para o blend com o modelo linear.

**Architecture:** A parte sem GPU (separar a dobra, montar as linhas de treino com o Open Pool, codificar rótulos) fica em `src/ntl/llm_data.py`, testada localmente. A parte com GPU (carregar em 4 bits, LoRA, treinar, prever) fica em `src/ntl/llm.py`, que só roda no Kaggle e tem um modo `SMOKE` para validar o caminho em minutos. `scripts/build_llm_notebook.py` monta o notebook a partir do pacote, como o `build_notebook.py` já faz.

**Tech Stack:** transformers, peft, bitsandbytes, torch (só no Kaggle); pandas, numpy, scikit-learn (local).

**Spec:** mensagem do usuário de 2026-11-07 ("melhorar a acurácia para pelo menos 0,700", sem GPU local, Kaggle/Colab) e a proposta de rodadas A–D aceita na conversa.

## Global Constraints

- Base: `Qwen/Qwen3-4B-Base`, carregado em NF4 com cálculo em fp16 (T4 não tem bf16).
- LoRA r=16, alpha=32, dropout 0,05 em `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`; tarefa `SEQ_CLS`, 18 rótulos na ordem de `ntl.labels.LABELS`.
- `max_length` 512 tokens, corte à direita; `pad_token = eos_token`.
- Treino: 1 época, lr 1e-4, cosseno, warmup 3%, batch 4, acumulação 4, seed 2027.
- Dados: train da competição + Open Pool limpo com 1.000 textos por rótulo (`ntl.pool.load_pool`), peso 0,85 no pool. O pool nunca entra na validação.
- Validação: dobra 0 do `ntl.validation.folds` (GroupKFold por prompt, 5 dobras, seed 2027). O linear é comparado na mesma dobra, a partir de `runs/oof_*.npy`.
- Limite do Kaggle: treino + validação abaixo de 9 h numa T4.
- Código sem comentários nem docstrings, commits curtos em português, sem trailer de coautoria.

## Review Focus

- Rótulo que falta na dobra de treino: a cabeça continua com 18 saídas e a probabilidade fica alinhada a `LABELS` — teste em `llm_data` (Task 1).
- Texto do pool idêntico a um texto da validação: vazaria — `training_rows` remove esses textos (Task 1).
- Texto vazio ou só espaços: a previsão não quebra — coberto no modo `SMOKE` (Task 2).
- Notebook gerado com célula `%%writefile` vazia: já quebrou uma vez; o teste do builder verifica (Task 3).
- Falta de memória na T4: `SMOKE` usa `max_length` 512 e batch 4, o mesmo pico do treino real (Task 2).

---

### Task 1: Dados da rodada sem GPU

**Files:**
- Create: `src/ntl/llm_data.py`
- Test: `tests/test_llm_data.py`

**Interfaces:**
- Consumes: `ntl.validation.folds(groups)`, `ntl.pool.load_pool(per_label)`, `ntl.labels.LABELS`.
- Produces:
  - `holdout(train: pd.DataFrame, fold: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]`
  - `label_ids(models: Iterable[str]) -> np.ndarray` (int64, índice em `LABELS`)
  - `training_rows(fit, valid, pool, pool_weight: float) -> pd.DataFrame` com colunas `text, label, weight`

- [ ] **Step 1: Write the failing tests** (usa o `train` sintético do `conftest`)

```python
def test_holdout_separa_prompts(train):
    fit, valid = holdout(train)
    assert set(fit["prompt_id"]).isdisjoint(valid["prompt_id"])
    assert len(fit) + len(valid) == len(train)

def test_rotulos_seguem_a_ordem_de_labels():
    assert label_ids(["gpt-5.5", "kimi-k3"]).tolist() == [0, 17]

def test_pool_entra_com_peso_e_sem_texto_da_validacao(train):
    fit, valid = holdout(train)
    pool = pd.DataFrame({"text": [valid["text"].iloc[0], "novo texto"], "model": ["gpt-5.5", "kimi-k3"]})
    rows = training_rows(fit, valid, pool, 0.85)
    assert valid["text"].iloc[0] not in set(rows["text"])
    assert rows.loc[rows["text"] == "novo texto", "weight"].item() == 0.85
    assert set(rows.columns) == {"text", "label", "weight"}
```

- [ ] **Step 2:** `.venv/Scripts/python -m pytest tests/test_llm_data.py -q` → FAIL (módulo não existe)
- [ ] **Step 3:** Implementar as três funções em `src/ntl/llm_data.py`.
- [ ] **Step 4:** mesmo comando → PASS; `ruff check .` limpo
- [ ] **Step 5:** `git commit -m "dados da rodada com LLM: dobra por prompt e pool sem vazamento"`

### Task 2: Treino e previsão com GPU

**Files:**
- Create: `src/ntl/llm.py`

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - `@dataclass(frozen=True) class LlmRun` com os valores das Global Constraints e `smoke: bool = False` (smoke: 64 textos de treino, 32 de validação, 8 passos)
  - `load(run) -> tuple[model, tokenizer]`
  - `train(run, model, tokenizer, rows) -> model` (subclasse de `Trainer` com `compute_loss` que pondera a cross-entropy por `weight`)
  - `predict_proba(model, tokenizer, texts, batch: int = 16) -> np.ndarray` (n × 18, ordem de `LABELS`, softmax em fp32)

- [ ] **Step 1:** Implementar `src/ntl/llm.py` (torch só neste módulo; os testes locais não importam ele).
- [ ] **Step 2:** `ast.parse` do arquivo sem erro; `ruff check .` limpo
- [ ] **Step 3:** `git commit -m "treino e previsão do Qwen com LoRA"`

### Task 3: Notebook da rodada

**Files:**
- Create: `scripts/build_llm_notebook.py`, `kaggle/llm_intro.md`
- Modify: `tests/test_notebook.py`

**Interfaces:**
- Consumes: Tasks 1–2; `build_notebook.cell`, `build_notebook.SETUP`, `build_notebook.MODULES`.
- Produces: `kaggle/llm_rodada_a.ipynb`. Célula final com `SMOKE = False`; treina na dobra 0, grava `/kaggle/working/proba_fold0.npy`, `/kaggle/working/metricas.json` (`balanced_accuracy`, `minutos`, `linhas_treino`) e o adaptador em `/kaggle/working/adaptador/`.

- [ ] **Step 1: Write the failing test**

```python
def test_notebook_da_rodada_a_leva_os_modulos_do_llm(monkeypatch):
    monkeypatch.chdir(ROOT)
    cells = build_llm_notebook.notebook("# intro")["cells"]
    sources = ["".join(c["source"]) for c in cells]
    assert any(s.startswith("%%writefile ntl/llm.py") for s in sources)
    assert "SMOKE = False" in sources[-1]
```

- [ ] **Step 2:** `pytest tests/test_notebook.py -q` → FAIL
- [ ] **Step 3:** Implementar o builder reaproveitando `cell`, `SETUP` e `MODULES` + `llm_data`, `llm`.
- [ ] **Step 4:** `pytest -q` → PASS; `python scripts/build_llm_notebook.py` gera o arquivo
- [ ] **Step 5:** `git commit -m "notebook da rodada A no Kaggle"`

### Task 4: Rodar e registrar

- [ ] **Step 1 (usuário):** subir o notebook, anexar o Open Pool e o `Qwen3-4B-Base` (Kaggle Models), GPU T4, `SMOKE = True` → termina sem erro em menos de 15 min.
- [ ] **Step 2 (usuário):** `SMOKE = False`, Save & Run All; baixar `metricas.json` e `proba_fold0.npy`.
- [ ] **Step 3 (agente):** comparar com o linear na mesma dobra e com o blend geométrico dos dois; registrar em `runs/experimentos.jsonl`.
- [ ] **Step 4:** `git commit -m "rodada A: Qwen3-4B na dobra 0"`
