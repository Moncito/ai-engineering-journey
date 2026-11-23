"""Phase 01 · Part C: charts with matplotlib.

Numbers convince engineers; charts convince everyone else.
Implement the two functions, then run:

    uv run python 01-python-for-ai/report.py

and open the PNGs in 01-python-for-ai/figures/. Embed them in your phase README.
"""


from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from analysis import clean, daily_cost, load_requests  # noqa: E402

FIGURES = Path(__file__).parent / "figures"


def plot_latency_hist(df: pd.DataFrame, path: Path) -> None:
    """Plot overlapping latency histograms for successful requests."""
    successful = df[df["status"] == "ok"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for model, group in successful.groupby("model"):
        ax.hist(
            group["latency_ms"],
            bins=50,
            alpha=0.5,
            label=model,
        )

    ax.set_title("LLM Request Latency by Model")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Number of Requests")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=120,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_daily_cost(cost: pd.Series, path: Path) -> None:
    """Plot total daily LLM API costs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        cost.index,
        cost.values,
        marker="o",
        linewidth=2,
    )

    ax.set_title("Daily LLM API Cost")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cost (USD)")

    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        path,
        dpi=120,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)

    df = clean(load_requests())

    plot_latency_hist(
        df,
        FIGURES / "latency_hist.png",
    )

    plot_daily_cost(
        daily_cost(df),
        FIGURES / "daily_cost.png",
    )

    print(f"saved charts to {FIGURES}")


if __name__ == "__main__":
    main()
