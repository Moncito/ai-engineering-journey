"""Spec for analysis.py and report.py. Do not edit; make them pass."""

import numpy as np
import pandas as pd
import pytest
from analysis import (
    clean,
    cost_share_by_model,
    daily_cost,
    error_rate_by_model,
    latency_percentiles,
    load_requests,
)


@pytest.fixture
def tiny() -> pd.DataFrame:
    """Hand-built log where every answer can be checked by hand."""
    return pd.DataFrame(
        {
            "request_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
            "timestamp": pd.to_datetime(
                [
                    "2026-10-01 09:00",
                    "2026-10-01 18:00",
                    "2026-10-02 10:00",
                    "2026-10-04 08:00",  # no requests on 10-03
                    "2026-10-04 09:00",
                    "2026-10-04 23:59",
                ]
            ),
            "model": ["a", "a", "a", "b", "b", "b"],
            "prompt_tokens": [100, 200, 300, 100, 100, 100],
            "completion_tokens": [10, 20, 0, 10, 0, 0],
            "latency_ms": [100.0, 200.0, 300.0, 1000.0, 30000.0, 2000.0],
            "status": ["ok", "ok", "error", "ok", "timeout", "error"],
            "cost_usd": [1.0, 2.0, 1.0, 3.0, 2.0, 1.0],
        }
    )


# --- load + clean ------------------------------------------------------------


def test_load_parses_timestamps():
    df = load_requests()
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert len(df) == 3015


def test_clean_real_dataset():
    raw = load_requests()
    df = clean(raw)
    assert len(df) == 2967
    assert not df.duplicated().any()
    assert (df["prompt_tokens"] >= 0).all()
    assert df["latency_ms"].notna().all()
    assert list(df.index) == list(range(len(df)))


def test_clean_does_not_modify_input(tiny):
    dirty = pd.concat([tiny, tiny.iloc[[0]]])
    dirty.loc[dirty.index[1], "latency_ms"] = np.nan
    before = dirty.copy()
    clean(dirty)
    pd.testing.assert_frame_equal(dirty, before)


def test_clean_drops_each_kind_of_dirt(tiny):
    dirty = pd.concat([tiny, tiny.iloc[[0]]], ignore_index=True)  # 1 duplicate
    dirty.loc[1, "prompt_tokens"] = -1  # r2 bad tokens
    dirty.loc[2, "latency_ms"] = np.nan  # r3 missing latency
    out = clean(dirty)
    assert list(out["request_id"]) == ["r1", "r4", "r5", "r6"]


# --- metrics -----------------------------------------------------------------


def test_error_rate_by_model(tiny):
    rates = error_rate_by_model(tiny)
    assert list(rates.index) == ["b", "a"]  # sorted highest first
    assert rates["a"] == pytest.approx(1 / 3)
    assert rates["b"] == pytest.approx(2 / 3)  # timeout counts as failure


def test_latency_percentiles_uses_only_ok(tiny):
    lat = latency_percentiles(tiny)
    assert list(lat.columns) == ["p50", "p95"]
    assert lat.loc["a", "p50"] == pytest.approx(150.0)  # ok latencies: 100, 200
    assert lat.loc["a", "p95"] == pytest.approx(195.0)
    assert lat.loc["b", "p50"] == pytest.approx(1000.0)  # only one ok request


def test_daily_cost(tiny):
    cost = daily_cost(tiny)
    assert list(cost.index) == list(pd.date_range("2026-10-01", "2026-10-04", freq="D"))
    assert list(cost.values) == [3.0, 1.0, 0.0, 6.0]


def test_cost_share_by_model(tiny):
    share = cost_share_by_model(tiny)
    assert list(share.index) == ["b", "a"]
    assert share.sum() == pytest.approx(100.0)
    assert share["b"] == pytest.approx(60.0)


@pytest.mark.parametrize(
    "fn", [error_rate_by_model, latency_percentiles, daily_cost, cost_share_by_model]
)
def test_metrics_do_not_modify_input(tiny, fn):
    before = tiny.copy()
    fn(tiny)
    pd.testing.assert_frame_equal(tiny, before)


# --- report ------------------------------------------------------------------


def test_report_writes_pngs(tiny, tmp_path):
    from report import plot_daily_cost, plot_latency_hist

    plot_latency_hist(tiny, tmp_path / "lat.png")
    plot_daily_cost(daily_cost(tiny), tmp_path / "cost.png")
    for name in ("lat.png", "cost.png"):
        f = tmp_path / name
        assert f.exists() and f.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
