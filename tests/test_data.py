"""Synthetic data generator: determinism and OHLCV sanity."""

from __future__ import annotations

from datetime import date

import pytest

from algotrade.data import SYMBOLS, generate_dataset, session_timestamps, trading_days


def test_generation_is_deterministic():
    d1 = generate_dataset(symbols=["RELIANCE"], n_days=5, seed=42)
    d2 = generate_dataset(symbols=["RELIANCE"], n_days=5, seed=42)
    pd_frame1 = d1["RELIANCE"]
    pd_frame2 = d2["RELIANCE"]
    assert pd_frame1.equals(pd_frame2)


def test_different_seeds_differ():
    d1 = generate_dataset(symbols=["RELIANCE"], n_days=5, seed=1)
    d2 = generate_dataset(symbols=["RELIANCE"], n_days=5, seed=2)
    assert not d1["RELIANCE"]["close"].equals(d2["RELIANCE"]["close"])


def test_ohlc_invariant_holds():
    data = generate_dataset(symbols=list(SYMBOLS), n_days=10, seed=42)
    for sym, df in data.items():
        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all(), sym
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all(), sym
        assert (df["high"] >= df["low"]).all(), sym
        assert (df["volume"] > 0).all(), sym
        assert (df["close"] > 0).all(), sym


def test_trading_days_skips_weekends():
    # 2024-01-06 is a Saturday
    days = trading_days(date(2024, 1, 5), 3)  # Fri, then skip weekend
    assert days[0] == date(2024, 1, 5)  # Friday
    assert days[1] == date(2024, 1, 8)  # Monday
    assert all(d.weekday() < 5 for d in days)


def test_session_timestamps_count_and_bounds():
    ts = session_timestamps(date(2024, 1, 2), bar_minutes=5)
    assert len(ts) == 375 // 5
    assert ts[0].time().isoformat(timespec="minutes") == "09:15"
    assert ts[-1].time().isoformat(timespec="minutes") == "15:25"


def test_session_timestamps_rejects_non_divisor():
    with pytest.raises(ValueError):
        session_timestamps(date(2024, 1, 2), bar_minutes=7)


def test_unknown_symbol_raises():
    with pytest.raises(KeyError):
        generate_dataset(symbols=["NOTASYMBOL"], n_days=2, seed=1)


def test_each_day_has_expected_bar_count():
    data = generate_dataset(symbols=["TCS"], n_days=4, bar_minutes=5, seed=42)
    df = data["TCS"]
    counts = df.groupby(df.index.date).size()
    assert (counts == 75).all()
