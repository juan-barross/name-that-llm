import pandas as pd

from ntl.llm_data import holdout, label_ids, training_rows


def test_holdout_separa_prompts(train):
    fit, valid = holdout(train)

    assert set(fit["prompt_id"]).isdisjoint(valid["prompt_id"])
    assert len(fit) + len(valid) == len(train)


def test_dobras_diferentes_dao_validacoes_diferentes(train):
    assert set(holdout(train, 0)[1]["prompt_id"]) != set(holdout(train, 1)[1]["prompt_id"])


def test_rotulos_seguem_a_ordem_de_labels():
    assert label_ids(["gpt-5.5", "kimi-k3"]).tolist() == [0, 17]


def test_pool_entra_com_peso_e_sem_texto_da_validacao(train):
    fit, valid = holdout(train)
    leaked = valid["text"].iloc[0]
    pool = pd.DataFrame({"text": [leaked, "novo texto"], "model": ["gpt-5.5", "kimi-k3"]})

    rows = training_rows(fit, valid, pool, 0.85)

    assert leaked not in set(rows["text"])
    assert rows.loc[rows["text"] == "novo texto", "weight"].item() == 0.85
    assert rows.loc[rows["text"] == "novo texto", "label"].item() == 17
    assert (rows["weight"].iloc[: len(fit)] == 1.0).all()
    assert list(rows.columns) == ["text", "label", "weight"]
