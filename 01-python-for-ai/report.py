"""Phase 01 · Part C: charts with matplotlib.

Numbers convince engineers; charts convince everyone else.
Implement the two functions, then run:

    uv run python 01-python-for-ai/report.py

and open the PNGs in 01-python-for-ai/figures/. Embed them in your phase README.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402, F401  (you'll use it)
import pandas as pd  # noqa: E402
from analysis import clean, daily_cost, load_requests  # noqa: E402

FIGURES = Path(__file__).parent / "figures"


def plot_latency_hist(df: pd.DataFrame, path: Path) -> None:
    """One histogram per model of latency_ms (status == "ok" only), overlaid on one axes.

    Requirements: title, x-label with units, legend, saved to `path`, figure closed after.
    Hints: fig, ax = plt.subplots(); ax.hist(values, bins=50, alpha=0.5, label=model)
           fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    """
    raise NotImplementedError


def plot_daily_cost(cost: pd.Series, path: Path) -> None:
    """Line chart of daily cost. Title, axis labels with units, saved to `path`, closed after."""
    raise NotImplementedError


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    df = clean(load_requests())
    plot_latency_hist(df, FIGURES / "latency_hist.png")
    plot_daily_cost(daily_cost(df), FIGURES / "daily_cost.png")
    print(f"saved charts to {FIGURES}")


if __name__ == "__main__":
    main()
