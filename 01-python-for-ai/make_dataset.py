"""Generate a synthetic LLM API request log: the kind of data an AI engineer analyzes daily.

Seeded, so it's reproducible. Includes realistic dirt on purpose (missing values,
bad rows, duplicates) because real logs are never clean.

    uv run python 01-python-for-ai/make_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "datasets" / "llm_requests.csv"

# Fictional models: (share of traffic, $ per 1K prompt tokens, $ per 1K completion tokens,
#                    base latency ms, ms per completion token, error rate)
MODELS = {
    "model-small": (0.55, 0.0002, 0.0008, 150, 4, 0.01),
    "model-medium": (0.35, 0.0010, 0.0040, 300, 9, 0.02),
    "model-large": (0.10, 0.0050, 0.0200, 600, 18, 0.05),
}


def generate(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(MODELS)
    shares = [MODELS[m][0] for m in names]

    model = rng.choice(names, size=n, p=shares)
    start = pd.Timestamp("2026-10-01")
    seconds = np.sort(rng.integers(0, 30 * 24 * 3600, size=n))
    timestamp = start + pd.to_timedelta(seconds, unit="s")

    prompt_tokens = rng.lognormal(mean=6.0, sigma=0.8, size=n).astype(int) + 10
    completion_tokens = rng.lognormal(mean=5.0, sigma=0.7, size=n).astype(int) + 1

    latency = np.empty(n)
    cost = np.empty(n)
    status = np.empty(n, dtype=object)
    for i, m in enumerate(model):
        _, p_in, p_out, base, per_tok, err = MODELS[m]
        latency[i] = base + per_tok * completion_tokens[i] + rng.gamma(2.0, 40.0)
        cost[i] = prompt_tokens[i] / 1000 * p_in + completion_tokens[i] / 1000 * p_out
        r = rng.random()
        status[i] = "error" if r < err else ("timeout" if r < err * 1.5 else "ok")
        if status[i] != "ok":
            completion_tokens[i] = 0
            cost[i] = prompt_tokens[i] / 1000 * p_in
        if status[i] == "timeout":
            latency[i] = 30_000.0

    df = pd.DataFrame(
        {
            "request_id": [f"req_{i:05d}" for i in range(n)],
            "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency.round(1),
            "status": status,
            "cost_usd": cost.round(6),
        }
    )

    # --- dirt ---------------------------------------------------------------
    df.loc[rng.choice(n, 25, replace=False), "latency_ms"] = np.nan  # logging gaps
    df.loc[rng.choice(n, 8, replace=False), "prompt_tokens"] = -1  # client bug
    dupes = df.sample(15, random_state=seed)  # retried log writes
    return pd.concat([df, dupes]).sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df):,} rows -> {OUT}")
