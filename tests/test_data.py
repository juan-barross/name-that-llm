import pandas as pd
import pytest

from ntl import data


def test_texto_vazio_vira_string_vazia(tmp_path):
    pd.DataFrame({"id": ["a", "b"], "text": ["oi", ""]}).to_csv(tmp_path / "test.csv", index=False)

    assert data.load("test", tmp_path)["text"].tolist() == ["oi", ""]


def test_sem_dados_falha_com_mensagem_clara(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "KAGGLE_DIR", tmp_path / "kaggle")
    monkeypatch.setattr(data, "LOCAL_DIR", tmp_path / "local")

    with pytest.raises(FileNotFoundError, match="train.csv"):
        data.data_dir()
