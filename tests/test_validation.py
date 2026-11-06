import numpy as np

from ntl.labels import LABELS
from ntl.models import EXPERIMENTS, build
from ntl.predict import aligned_probabilities
from ntl.validation import cross_validate, folds


def test_nenhum_prompt_aparece_no_treino_e_na_validacao(train):
    for fit_index, valid_index in folds(train["prompt_id"]):
        fit_prompts = set(train["prompt_id"].iloc[fit_index])
        assert fit_prompts.isdisjoint(train["prompt_id"].iloc[valid_index])


def test_cada_linha_e_validada_uma_vez(train):
    seen = np.concatenate([valid_index for _, valid_index in folds(train["prompt_id"])])
    assert sorted(seen) == list(range(len(train)))


def test_cv_devolve_uma_distribuicao_por_linha(train):
    result = cross_validate("char_word_logreg", train)

    assert result.probabilities.shape == (len(train), len(LABELS))
    assert np.allclose(result.probabilities.sum(axis=1), 1)
    assert len(result.fold_scores) == 5


def test_rotulo_ausente_no_treino_vira_coluna_zerada(train):
    kept = train[train["model"] != LABELS[-1]]
    model = build("char_logreg").fit(kept["text"], kept["model"])

    probabilities = aligned_probabilities(model, train["text"][:3])

    assert probabilities.shape == (3, len(LABELS))
    assert (probabilities[:, -1] == 0).all()
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_experimentos_montam_as_partes_declaradas():
    for name, experiment in EXPERIMENTS.items():
        pipeline = build(name)
        assert [part for part, _ in pipeline.named_steps["features"].transformer_list] == list(
            experiment.parts
        )
        strength = "alpha" if experiment.classifier == "sgd" else "C"
        assert pipeline.get_params()[f"classifier__{strength}"] == experiment.strength
