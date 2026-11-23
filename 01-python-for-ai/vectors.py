
"""Phase 01 · Part A: vectors with numpy."""

import time

import numpy as np


def dot_loop(a: list[float], b: list[float]) -> float:
    """Calculate the dot product using plain Python."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length.")

    result = 0.0

    for i in range(len(a)):
        result += a[i] * b[i]

    return result


def dot_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the dot product using NumPy."""
    return float(np.dot(a, b))


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(x)

    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")

    return x / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    a_normalized = normalize(a)
    b_normalized = normalize(b)

    return float(np.dot(a_normalized, b_normalized))


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert logits into probabilities."""
    shifted = logits - np.max(logits)

    exponentials = np.exp(shifted)

    return exponentials / np.sum(exponentials)


def pairwise_cosine(matrix: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between all vector pairs."""
    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    normalized = matrix / norms

    return normalized @ normalized.T


def benchmark(n: int = 1_000_000) -> None:
    """Compare Python and NumPy dot-product performance."""
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
