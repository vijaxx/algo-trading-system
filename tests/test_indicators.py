"""Indicator correctness against hand-computed values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algotrade.indicators import (
    atr,
    bollinger_bands,
    ema,
    rsi,
    sma,
    supertrend,
    true_range,
    vwap,
    wilder_smooth,
)


def test_sma_hand_computed():
    s = pd.Series([1, 2, 3, 4, 5, 6], dtype=float)
    out = sma(s, 3)
    # first 2 are NaN, then rolling means of window 3
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(2.0)  # (1+2+3)/3
    assert out.iloc[3] == pytest.approx(3.0)  # (2+3+4)/3
    assert out.iloc[5] == pytest.approx(5.0)  # (4+5+6)/3


def test_ema_matches_manual_recursion():
    s = pd.Series([10, 11, 12, 11, 13], dtype=float)
    period = 3
    alpha = 2 / (period + 1)
    manual = [10.0]
    for x in s.iloc[1:]:
        manual.append(alpha * x + (1 - alpha) * manual[-1])
    out = ema(s, period)
    np.testing.assert_allclose(out.to_numpy(), manual, rtol=1e-9)


def test_ema_first_value_equals_first_price():
    s = pd.Series([100.0, 105.0, 103.0])
    assert ema(s, 5).iloc[0] == pytest.approx(100.0)


def test_rolling_std_population_not_sample():
    s = pd.Series([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)
    # population std of the whole series (period = len) is a known value: 2.0
    out = sma(s, 8)  # sanity: full-window sma
    from algotrade.indicators import rolling_std

    sd = rolling_std(s, 8)
    assert sd.iloc[-1] == pytest.approx(2.0, abs=1e-9)


def test_rsi_all_up_moves_is_100():
    s = pd.Series([100 + i for i in range(20)], dtype=float)
    out = rsi(s, period=14)
    assert out.iloc[-1] == pytest.approx(100.0)


def test_rsi_all_down_moves_is_0():
    s = pd.Series([100 - i for i in range(20)], dtype=float)
    out = rsi(s, period=14)
    assert out.iloc[-1] == pytest.approx(0.0)


def test_rsi_bounded_0_100():
    rng = np.random.default_rng(0)
    s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    out = rsi(s, 14).dropna()
    assert (out >= 0).all() and (out <= 100).all()


def test_true_range_first_bar_is_high_minus_low():
    high = pd.Series([105.0, 108.0])
    low = pd.Series([100.0, 104.0])
    close = pd.Series([102.0, 107.0])
    tr = true_range(high, low, close)
    assert tr.iloc[0] == pytest.approx(5.0)


def test_true_range_uses_prev_close_when_gap():
    # gap up: prev close 100, today high 101 low 99 -> range small but
    # |high - prev_close| = 1... use a bigger gap to make it obvious
    high = pd.Series([100.0, 130.0])
    low = pd.Series([98.0, 125.0])
    close = pd.Series([99.0, 128.0])
    tr = true_range(high, low, close)
    # bar2: H-L=5, |H-prevC|=|130-99|=31, |L-prevC|=|125-99|=26 -> max=31
    assert tr.iloc[1] == pytest.approx(31.0)


def test_wilder_smooth_seed_is_simple_mean():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7], dtype=float)
    out = wilder_smooth(s, 4)
    assert out.iloc[3] == pytest.approx(2.5)  # mean(1,2,3,4)
    # next: (2.5*3 + 5)/4 = 3.125
    assert out.iloc[4] == pytest.approx(3.125)


def test_bollinger_bands_hand_computed():
    s = pd.Series([10, 12, 14, 12, 10], dtype=float)
    bands = bollinger_bands(s, period=5, num_std=2.0)
    mean = s.mean()
    sd = s.std(ddof=0)
    assert bands["mid"].iloc[-1] == pytest.approx(mean)
    assert bands["upper"].iloc[-1] == pytest.approx(mean + 2 * sd)
    assert bands["lower"].iloc[-1] == pytest.approx(mean - 2 * sd)


def test_bollinger_upper_always_gte_lower():
    rng = np.random.default_rng(1)
    s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 100)))
    bands = bollinger_bands(s, 20, 2.0).dropna()
    assert (bands["upper"] >= bands["lower"]).all()


def test_vwap_hand_computed():
    high = pd.Series([10.0, 11.0])
    low = pd.Series([9.0, 10.0])
    close = pd.Series([9.5, 10.5])
    volume = pd.Series([100, 200])
    out = vwap(high, low, close, volume)
    tp1 = (10 + 9 + 9.5) / 3
    tp2 = (11 + 10 + 10.5) / 3
    expected2 = (tp1 * 100 + tp2 * 200) / 300
    assert out.iloc[0] == pytest.approx(tp1)
    assert out.iloc[1] == pytest.approx(expected2)


def test_atr_positive_and_reasonable():
    rng = np.random.default_rng(2)
    n = 50
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    high = close + rng.uniform(0.5, 1.5, n)
    low = close - rng.uniform(0.5, 1.5, n)
    a = atr(high, low, close, 14).dropna()
    assert (a > 0).all()


def test_supertrend_direction_only_plus_or_minus_one():
    rng = np.random.default_rng(3)
    n = 80
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    high = close + rng.uniform(0.2, 1.0, n)
    low = close - rng.uniform(0.2, 1.0, n)
    st = supertrend(high, low, close, period=10, multiplier=3.0)
    valid = st["direction"].dropna()
    assert set(valid.unique()).issubset({1.0, -1.0})


def test_supertrend_uptrend_flags_long():
    # A strongly, monotonically rising series should end in an uptrend.
    close = pd.Series(np.linspace(100, 200, 60))
    high = close + 0.5
    low = close - 0.5
    st = supertrend(high, low, close, period=10, multiplier=3.0)
    assert st["direction"].iloc[-1] == 1.0


def test_indicators_reject_nonpositive_period():
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        sma(s, 0)
    with pytest.raises(ValueError):
        ema(s, -1)
