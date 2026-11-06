import numpy as np

from ntl.blend import blend_score, geometric_blend, search_weights
from ntl.labels import LABELS

LABEL_ARRAY = np.array(LABELS)


def member(correct_rows: np.ndarray, rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probabilities = rng.dirichlet(np.ones(len(LABELS)), size=rows)
    truth = np.arange(rows) % len(LABELS)
    probabilities[correct_rows, truth[correct_rows]] += 2
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def test_blend_de_um_membro_com_peso_um_e_o_proprio_membro():
    probabilities = member(np.arange(10), 36, seed=0)

    assert np.allclose(geometric_blend([probabilities], [1.0]), probabilities)


def test_membros_que_acertam_partes_diferentes_somam():
    rows = 72
    labels = LABEL_ARRAY[np.arange(rows) % len(LABELS)]
    first, second = member(np.arange(0, 36), rows, 1), member(np.arange(36, 72), rows, 2)

    weights, score = search_weights([first, second], labels)

    assert all(w > 0 for w in weights)
    assert score > max(blend_score([first], [1.0], labels), blend_score([second], [1.0], labels))


def test_membro_inutil_fica_com_peso_zero():
    rows = 72
    labels = LABEL_ARRAY[np.arange(rows) % len(LABELS)]
    good, noise = member(np.arange(rows), rows, 3), member(np.array([], dtype=int), rows, 4)

    weights, _ = search_weights([good, noise], labels)

    assert weights == [1.0, 0.0]
