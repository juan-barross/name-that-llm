import numpy as np

from ntl.features import MARKS, STATS, function_skeleton, style_matrix, text_style, vectorizer, word_shape


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


def test_forma_troca_palavras_por_maiuscula_ou_minuscula():
    assert word_shape("Hello world, 42 times!") == "X x, 42 x!"


def test_esqueleto_guarda_palavras_funcionais_e_pontuacao():
    assert function_skeleton("The cat sat on the mat.") == "the CONTENT CONTENT on the CONTENT ."


def test_visoes_de_estilo_ignoram_o_assunto():
    texts = ["The ocean is deep, and it is cold.", "The desert is dry, and it is hot."]
    shape = vectorizer("shape", "function").fit_transform(texts)

    assert (shape[0] != shape[1]).nnz == 0


def test_partes_do_vetorizador_seguem_o_pedido():
    assert [name for name, _ in vectorizer("char", "style").transformer_list] == ["char", "style"]
