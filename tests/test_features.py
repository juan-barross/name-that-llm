import numpy as np

from ntl.features import MARKS, style_matrix, text_style


def test_conta_travessao_por_caractere():
    features = dict(zip(MARKS, text_style("a — b — c"), strict=False))

    assert features["em_dash"] == 2 / 9


def test_texto_vazio_nao_quebra():
    assert np.isfinite(style_matrix([""])).all()


def test_uma_linha_por_texto():
    assert style_matrix(["um texto.", "outro texto!"]).shape[0] == 2
