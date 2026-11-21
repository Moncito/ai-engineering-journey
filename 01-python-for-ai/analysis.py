"""Phase 01 · Part B: analyzing an LLM API request log with pandas.

Questions an AI engineer gets asked every week:
  - Which model fails most?             -> error_rate_by_model
  - How slow is each model, really?     -> latency_percentiles (p95, not average!)
  - What are we spending per day?       -> daily_cost
  - Which model eats the budget?        -> cost_share_by_model

RULES: use pandas operations (boolean masks, groupby, resample). No for-loops over rows.
Every function must NOT modify the DataFrame passed in. Return new objects.

Run tests:  uv run pytest 01-python-for-ai -k analysis
Run it:     uv run python 01-python-for-ai/analysis.py
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "datasets" / "llm_requests.csv"


def load_requests(path: Path = DATA) -> pd.DataFrame:
    """Read the CSV. The `timestamp` column must be parsed as datetime64, not text.

    Hint: pd.read_csv(..., parse_dates=[...])
    """
    raise NotImplementedError


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy:
      1. drop exact duplicate rows
      2. drop rows where prompt_tokens < 0     (client bug)
      3. drop rows where latency_ms is missing (logging gap)
    and reset the index (0, 1, 2, ...).

    Hints: .drop_duplicates(), boolean mask df[df["col"] >= 0], .dropna(subset=[...]),
           .reset_index(drop=True)
    """
    raise NotImplementedError


def error_rate_by_model(df: pd.DataFrame) -> pd.Series:
    """Fraction of requests per model whose status is NOT "ok" (errors AND timeouts).

    Return a Series indexed by model name, values in [0, 1], sorted highest first.

    Hint: build a boolean column, then groupby("model") and .mean(). The mean of
          True/False is the fraction of True.
    """
    raise NotImplementedError


def latency_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """p50 and p95 latency per model, using ONLY successful (status == "ok") requests.

    Return a DataFrame indexed by model with exactly two columns: "p50" and "p95".

    Why p95: averages hide the slow tail. "p95 = 3s" means 1 in 20 users waits 3s+.
    Hint: groupby("model")["latency_ms"].quantile(...) or .agg(p50=..., p95=...)
    """
    raise NotImplementedError


def daily_cost(df: pd.DataFrame) -> pd.Series:
    """Total cost_usd per calendar day.

    Return a Series indexed by day (DatetimeIndex at midnight), sorted by date.
    Days with no requests inside the range should appear with 0.0.

    Hint: df.set_index("timestamp")["cost_usd"].resample("D").sum()
    """
    raise NotImplementedError


def cost_share_by_model(df: pd.DataFrame) -> pd.Series:
    """Percentage of total cost per model (values sum to 100), sorted highest first."""
    raise NotImplementedError


def main() -> None:
    raw = load_requests()
    df = clean(raw)
    print(f"rows: {len(raw):,} raw -> {len(df):,} clean ({len(raw) - len(df)} dropped)\n")
    print("Error rate by model:")
    print(error_rate_by_model(df).map("{:.1%}".format), "\n")
    print("Latency (ms), successful requests:")
    print(latency_percentiles(df).round(0), "\n")
    print("Cost share by model:")
    print(cost_share_by_model(df).map("{:.1f}%".format), "\n")
    cost = daily_cost(df)
    print(f"Total cost: ${cost.sum():.2f} over {len(cost)} days (${cost.mean():.3f}/day avg)")


if __name__ == "__main__":
    main()
