
"""Phase 01 · Part B: analyzing an LLM API request log with pandas."""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "datasets" / "llm_requests.csv"


def load_requests(path: Path = DATA) -> pd.DataFrame:
    """Read CSV and parse timestamp as datetime."""
    return pd.read_csv(
        path,
        parse_dates=["timestamp"],
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean request logs without modifying the original DataFrame."""
    cleaned = df.drop_duplicates()

    cleaned = cleaned[cleaned["prompt_tokens"] >= 0]

    cleaned = cleaned.dropna(subset=["latency_ms"])

    return cleaned.reset_index(drop=True)


def error_rate_by_model(df: pd.DataFrame) -> pd.Series:
    """Calculate error and timeout rates by model."""
    is_error = df["status"] != "ok"

    result = is_error.groupby(df["model"]).mean()

    return result.sort_values(ascending=False)


def latency_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate p50 and p95 latency for successful requests."""
    successful = df[df["status"] == "ok"]

    grouped = successful.groupby("model")["latency_ms"]

    result = grouped.quantile([0.50, 0.95]).unstack()

    return result.rename(
        columns={0.50: "p50", 0.95: "p95"}
    ).reindex(columns=["p50", "p95"])


def daily_cost(df: pd.DataFrame) -> pd.Series:
    """Calculate total API costs per calendar day."""
    return (
        df.set_index("timestamp")["cost_usd"]
        .resample("D")
        .sum()
    )


def cost_share_by_model(df: pd.DataFrame) -> pd.Series:
    """Calculate percentage of API spending by model."""
    costs = df.groupby("model")["cost_usd"].sum()

    total = costs.sum()

    if total == 0:
        return (costs * 0.0).sort_values(ascending=False)

    percentages = (costs / total) * 100

    return percentages.sort_values(ascending=False)


def main() -> None:
    raw = load_requests()
    df = clean(raw)

    print(
        f"rows: {len(raw):,} raw -> {len(df):,} clean "
        f"({len(raw) - len(df)} dropped)\n"
    )

    print("Error rate by model:")
    print(error_rate_by_model(df).map("{:.1%}".format), "\n")

    print("Latency (ms), successful requests:")
    print(latency_percentiles(df).round(0), "\n")

    print("Cost share by model:")
    print(cost_share_by_model(df).map("{:.1f}%".format), "\n")

    cost = daily_cost(df)

    print(
        f"Total cost: ${cost.sum():.2f} over {len(cost)} days "
        f"(${cost.mean():.3f}/day avg)"
    )


if __name__ == "__main__":
    main()
