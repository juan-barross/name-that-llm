import numpy as np

from ntl.prior import sinkhorn, soft_balance


def skewed(rows: int = 60, classes: int = 3, seed: int = 0) -> np.ndarray:
    logits = np.random.default_rng(seed).normal(size=(rows, classes))
    logits[:, 0] += 2
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def test_sinkhorn_iguala_a_massa_de_cada_classe():
    balanced = sinkhorn(skewed())

    assert np.allclose(balanced.sum(axis=1), 1)
    assert np.allclose(balanced.sum(axis=0), 20, atol=0.5)


def test_peso_zero_nao_muda_nada():
    probabilities = skewed()

    assert soft_balance(probabilities, 0.0) is probabilities


def test_balanceamento_suave_fica_entre_o_original_e_o_total():
    probabilities = skewed()
    original, half, full = (soft_balance(probabilities, w)[:, 0].sum() for w in (0.0, 0.5, 1.0))

    assert full < half < original
    assert np.allclose(soft_balance(probabilities, 0.5).sum(axis=1), 1)
