import numpy as np

from ntl.features import MARKS, STATS, style_matrix, text_style, vectorizer


def named_style(text: str) -> dict[str, float]:
    return dict(zip([*MARKS, *STATS], text_style(text), strict=True))


def test_conta_travessao_por_caractere():
    assert named_style("a — b — c")["em_dash"] == 2 / 9


def test_frases_medidas_em_palavras():
    style = named_style("Uma frase curta. Outra frase bem mais longa aqui.")

    assert style["mean_sentence_words"] == 4.5


def test_texto_vazio_nao_quebra():
    assert np.isfinite(style_matrix([""])).all()


def test_uma_linha_por_texto_e_uma_coluna_por_marca():
    assert style_matrix(["um texto.", "outro texto!"]).shape == (2, len(MARKS) + len(STATS))


def test_partes_do_vetorizador_seguem_o_pedido():
    assert [name for name, _ in vectorizer("char", "style").transformer_list] == ["char", "style"]
