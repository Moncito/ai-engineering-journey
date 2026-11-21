"""Spec for vectors.py. Do not edit; make them pass."""

import inspect

import numpy as np
import pytest
import vectors
from vectors import (
    cosine_similarity,
    dot_loop,
    dot_numpy,
    normalize,
    pairwise_cosine,
    softmax,
)

# --- dot -------------------------------------------------------------------


def test_dot_loop_basic():
    assert dot_loop([1, 2, 3], [4, 5, 6]) == 32


def test_dot_loop_length_mismatch():
    with pytest.raises(ValueError):
        dot_loop([1, 2], [1, 2, 3])


def test_dot_loop_does_not_use_numpy():
    src = inspect.getsource(vectors.dot_loop)
    assert "np." not in src, "dot_loop is the plain-Python baseline: no numpy"


def test_dot_numpy_matches_loop():
    rng = np.random.default_rng(1)
    a, b = rng.random(100), rng.random(100)
    assert dot_numpy(a, b) == pytest.approx(dot_loop(a.tolist(), b.tolist()))


def test_dot_numpy_returns_python_float():
    assert type(dot_numpy(np.array([1.0, 2.0]), np.array([3.0, 4.0]))) is float


# --- normalize -------------------------------------------------------------


def test_normalize_length_is_one():
    v = normalize(np.array([3.0, 4.0]))
    np.testing.assert_allclose(v, [0.6, 0.8])
    assert np.linalg.norm(v) == pytest.approx(1.0)


def test_normalize_zero_vector_raises():
    with pytest.raises(ValueError):
        normalize(np.zeros(3))


# --- cosine ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1, 0], [5, 0], 1.0),  # same direction, different length
        ([1, 0], [0, 3], 0.0),  # orthogonal
        ([1, 2], [-1, -2], -1.0),  # opposite
        ([1, 1], [1, 0], 2**-0.5),  # 45 degrees
    ],
)
def test_cosine_similarity(a, b, expected):
    got = cosine_similarity(np.array(a, dtype=float), np.array(b, dtype=float))
    assert got == pytest.approx(expected)
    assert type(got) is float


# --- softmax ---------------------------------------------------------------


def test_softmax_sums_to_one_and_positive():
    p = softmax(np.array([2.0, 1.0, 0.1]))
    assert p.sum() == pytest.approx(1.0)
    assert (p > 0).all()
    np.testing.assert_allclose(p, [0.659001, 0.242433, 0.098566], atol=1e-6)


def test_softmax_keeps_order():
    p = softmax(np.array([0.5, 3.0, -1.0]))
    assert p.argmax() == 1 and p.argmin() == 2


def test_softmax_is_numerically_stable():
    p = softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert not np.isnan(p).any(), "overflow: subtract the max before exp"
    np.testing.assert_allclose(p, softmax(np.array([0.0, 1.0, 2.0])))


def test_softmax_uniform():
    np.testing.assert_allclose(softmax(np.zeros(4)), [0.25] * 4)


# --- pairwise --------------------------------------------------------------


def test_pairwise_cosine_shape_and_diagonal():
    rng = np.random.default_rng(2)
    m = rng.normal(size=(5, 8))
    sims = pairwise_cosine(m)
    assert sims.shape == (5, 5)
    np.testing.assert_allclose(np.diag(sims), np.ones(5))
    np.testing.assert_allclose(sims, sims.T)  # symmetric


def test_pairwise_cosine_matches_single():
    rng = np.random.default_rng(3)
    m = rng.normal(size=(4, 6))
    sims = pairwise_cosine(m)
    assert sims[1, 3] == pytest.approx(cosine_similarity(m[1], m[3]))


def test_pairwise_cosine_has_no_python_loops():
    src = inspect.getsource(vectors.pairwise_cosine)
    body = src.split('"""')[-1]  # skip the docstring
    assert "for " not in body and "while " not in body, "vectorize it: no loops"
