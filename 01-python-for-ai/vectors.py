"""Phase 01 · Part A: vectors with numpy.

Every LLM, embedding model, and neural net is built from these few operations.
Later phases depend on them:
  - cosine_similarity  -> RAG retrieval (Phase 07)
  - softmax            -> how an LLM turns scores into next-token probabilities (Phase 05)
  - pairwise_cosine    -> "find the most similar documents" in one matrix multiply

RULES:
  - dot_loop: plain Python only (no numpy). It's the slow baseline.
  - everything else: numpy, NO Python for-loops. Use vectorized operations.

Run tests:      uv run pytest 01-python-for-ai -k vectors
Run benchmark:  uv run python 01-python-for-ai/vectors.py
"""

import time

import numpy as np


def dot_loop(a: list[float], b: list[float]) -> float:
    """Dot product with a plain Python for-loop: sum of a[i] * b[i].

    Raise ValueError if the lengths differ.
    """
    raise NotImplementedError


def dot_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Same dot product, vectorized. One line. Return a Python float."""
    raise NotImplementedError


def normalize(x: np.ndarray) -> np.ndarray:
    """Return x scaled to length 1 (divide by its L2 norm).

    Raise ValueError for an all-zero vector (it has no direction).
    Hint: np.linalg.norm
    """
    raise NotImplementedError


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle between a and b, a float in [-1, 1].

        1.0  -> same direction (very similar meaning, for embeddings)
        0.0  -> unrelated (orthogonal)
       -1.0  -> opposite

    Hint: reuse normalize + dot. Return a Python float.
    """
    raise NotImplementedError


def softmax(logits: np.ndarray) -> np.ndarray:
    """Turn raw scores into probabilities that are all > 0 and sum to 1.

        softmax(x)_i = exp(x_i) / sum_j exp(x_j)

    Must be numerically stable: subtract max(logits) before exp,
    otherwise np.exp(1000) overflows to inf and you get nan.
    Works on a 1-D array.
    """
    raise NotImplementedError


def pairwise_cosine(matrix: np.ndarray) -> np.ndarray:
    """Given an (n, d) matrix where each row is a vector, return the (n, n) matrix
    of cosine similarities between every pair of rows.

    No loops. Hints:
      - normalize every row at once: np.linalg.norm(matrix, axis=1, keepdims=True)
      - then a single matrix multiply: unit @ unit.T
    You may assume no row is all zeros.
    """
    raise NotImplementedError


def benchmark(n: int = 1_000_000) -> None:
    """Time dot_loop vs dot_numpy on n-element vectors."""
    rng = np.random.default_rng(0)
    a, b = rng.random(n), rng.random(n)
    a_list, b_list = a.tolist(), b.tolist()

    t0 = time.perf_counter()
    slow = dot_loop(a_list, b_list)
    t1 = time.perf_counter()
    fast = dot_numpy(a, b)
    t2 = time.perf_counter()

    print(f"n = {n:,}")
    print(f"dot_loop  : {t1 - t0:8.4f} s  -> {slow:.4f}")
    print(f"dot_numpy : {t2 - t1:8.4f} s  -> {fast:.4f}")
    print(f"speedup   : {(t1 - t0) / (t2 - t1):,.0f}x")


if __name__ == "__main__":
    benchmark()
