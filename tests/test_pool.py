import pandas as pd
import pytest

from ntl import models
from ntl.models import Experiment, training_frame
from ntl.pool import RISKS, clean, sample
from ntl.validation import cross_validate


def raw_pool(**overrides) -> pd.DataFrame:
    row = {"text": "texto do pool", "model": "kimi-k3", "n_words": 100, "match_weight": 1.0} | dict.fromkeys(
        RISKS, False
    )
    return pd.DataFrame([row | overrides])


@pytest.mark.parametrize("risk", RISKS)
def test_tira_texto_com_risco_de_vazamento(risk):
    assert clean(raw_pool(**{risk: True})).empty


@pytest.mark.parametrize("overrides", [{"n_words": 20}, {"n_words": 400}, {"match_weight": 0.0}])
def test_tira_texto_fora_do_formato_do_treino(overrides):
    assert clean(raw_pool(**overrides)).empty


def test_mantem_texto_limpo():
    assert len(clean(raw_pool())) == 1


def test_amostra_no_maximo_n_por_modelo_e_e_reproduzivel():
    pool = pd.concat([raw_pool(model="kimi-k3")] * 5 + [raw_pool(model="glm-5.3")] * 2, ignore_index=True)
    pool["text"] = [f"t{i}" for i in range(len(pool))]

    first, second = sample(pool, 3), sample(pool, 3)

    assert first["model"].value_counts().to_dict() == {"kimi-k3": 3, "glm-5.3": 2}
    assert first["text"].tolist() == second["text"].tolist()


@pytest.fixture
def with_pool(monkeypatch, train):
    pool = train[["text", "model"]].head(36).assign(text=lambda t: "pool " + t["text"])
    monkeypatch.setitem(
        models.EXPERIMENTS, "pool_test", Experiment(("char",), pool_per_label=2, pool_weight=0.5)
    )
    monkeypatch.setattr(models, "load_pool", lambda per_label: pool)
    return pool


def test_pool_entra_no_treino_com_peso_proprio(train, with_pool):
    data = training_frame("pool_test", train)

    assert len(data) == len(train) + len(with_pool)
    assert data["weight"].value_counts().to_dict() == {1.0: len(train), 0.5: len(with_pool)}


def test_pool_nunca_entra_na_validacao(train, with_pool):
    result = cross_validate("pool_test", train)

    assert result.probabilities.shape[0] == len(train)
