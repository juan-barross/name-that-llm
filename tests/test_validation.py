import numpy as np

from ntl.models import EXPERIMENTS, build
from ntl.validation import cross_validate, folds


def test_nenhum_prompt_aparece_no_treino_e_na_validacao(train):
    for fit_index, valid_index in folds(train["prompt_id"]):
        fit_prompts = set(train["prompt_id"].iloc[fit_index])
        assert fit_prompts.isdisjoint(train["prompt_id"].iloc[valid_index])


def test_cada_linha_e_validada_uma_vez(train):
    seen = np.concatenate([valid_index for _, valid_index in folds(train["prompt_id"])])
    assert sorted(seen) == list(range(len(train)))


def test_cv_aprende_o_estilo_de_cada_modelo(train):
    result = cross_validate(lambda: build("char_word_logreg"), train)

    assert result.probabilities.shape == (len(train), 18)
    assert np.allclose(result.probabilities.sum(axis=1), 1)
    assert result.score > 0.9


def test_todos_os_experimentos_montam(train):
    for name in EXPERIMENTS:
        assert build(name).fit(train["text"], train["model"]).predict(train["text"][:2]).shape == (2,)
