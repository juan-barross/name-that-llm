import pandas as pd
import pytest

from ntl.labels import LABELS
from ntl.predict import fit_predict
from ntl.submission import write


def test_grava_parquet_valido(tmp_path):
    path = tmp_path / "submission.parquet"

    write(pd.Series(["a", "b"]), ["gpt-5.5", "kimi-k3"], path)

    assert pd.read_parquet(path).to_dict("list") == {"id": ["a", "b"], "model": ["gpt-5.5", "kimi-k3"]}


@pytest.mark.parametrize(
    ("ids", "labels", "message"),
    [
        (["a", "a"], ["gpt-5.5", "gpt-5.5"], "repetido"),
        (["a"], ["gpt-4"], "fora da lista"),
        (["a"], [None], "nulo"),
    ],
)
def test_recusa_submissao_invalida(tmp_path, ids, labels, message):
    with pytest.raises(ValueError, match=message):
        write(pd.Series(ids), labels, tmp_path / "s.parquet")


def test_teste_de_uma_linha_nova_como_na_sessao_interativa(train):
    test = pd.DataFrame({"id": ["novo"], "text": ["Sobre oceanos, marca3 porém logo marca3."]})

    prediction = fit_predict({"char_logreg": 1.0, "char_word_logreg": 0.5}, train, test, balance=0.3)

    assert prediction.shape == (1,)
    assert prediction[0] in LABELS
