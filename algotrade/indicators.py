"""Technical indicators implemented from scratch with pandas/numpy.

Deliberately no TA-Lib dependency: TA-Lib needs a compiled C library, which
makes `pip install -r requirements.txt` fail on a clean machine. Everything
here is pure pandas/numpy so the repo runs after a plain clone.

Conventions used throughout:
  * Every function takes and returns pandas Series/DataFrame aligned on the
    input index.
  * "Wilder smoothing" (used by RSI / ATR / Supertrend) is the recursive
    average from J. Welles Wilder's *New Concepts in Technical Trading
    Systems* (1978):  avg_t = (avg_{t-1} * (n - 1) + x_t) / n, seeded with a
    simple mean of the first n observations.
  * Standard deviation for Bollinger Bands uses ddof=0 (population), which is
    what charting platforms (TradingView, Zerodha Kite) use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    _check_period(period)
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average, alpha = 2 / (period + 1), recursive form.

    Uses adjust=False so the first value equals the first price and each
    subsequent value is  ema_t = alpha * x_t + (1 - alpha) * ema_{t-1}.
    That recursive form is what a live trading system can compute
    incrementally, and it matches broker charting defaults.
    """
    _check_period(period)
    return series.ewm(span=period, adjust=False).mean()


def rolling_std(series: pd.Series, period: int) -> pd.Series:
    """Population standard deviation (ddof=0) over a rolling window."""
    _check_period(period)
    return series.rolling(window=period, min_periods=period).std(ddof=0)


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's recursive moving average (a.k.a. RMA / SMMA).

    Seeded with the simple mean of the first `period` values; NaN before that.
    """
    _check_period(period)
    values = series.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    n = len(values)
    if n < period:
        return pd.Series(out, index=series.index)

    seed = np.nanmean(values[:period])
    out[period - 1] = seed
    prev = seed
    for i in range(period, n):
        prev = (prev * (period - 1) + values[i]) / period
        out[i] = prev
    return pd.Series(out, index=series.index)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder smoothing.

    RSI = 100 - 100 / (1 + avg_gain / avg_loss).
    When avg_loss == 0 the RSI is defined as 100 (no downside at all).
    The first valid value appears at index `period` (needs `period` deltas).
    """
    _check_period(period)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # drop the leading NaN from .diff() before seeding the Wilder average so
    # the seed is the mean of the first `period` *deltas*, not of period-1.
    avg_gain = wilder_smooth(gain.iloc[1:], period).reindex(close.index)
    avg_loss = wilder_smooth(loss.iloc[1:], period).reindex(close.index)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 with a valid avg_gain -> pure uptrend -> RSI 100
    out = out.where(~((avg_loss == 0.0) & avg_gain.notna()), 100.0)
    # both zero (perfectly flat) -> neutral 50
    out = out.where(~((avg_loss == 0.0) & (avg_gain == 0.0)), 50.0)
    return out


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range = max(H-L, |H - prev_close|, |L - prev_close|).

    The first bar has no previous close, so TR falls back to H-L.
    """
    prev_close = close.shift(1)
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    tr = pd.concat([a, b, c], axis=1).max(axis=1)
    tr.iloc[0] = (high.iloc[0] - low.iloc[0]) if len(high) else np.nan
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothed True Range)."""
    return wilder_smooth(true_range(high, low, close), period)


def bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands: SMA +/- num_std * population stdev.

    Returns a DataFrame with columns: mid, upper, lower, bandwidth.
    `bandwidth` = (upper - lower) / mid, the standard squeeze measure.
    """
    mid = sma(close, period)
    sd = rolling_std(close, period)
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    return pd.DataFrame(
        {
            "mid": mid,
            "upper": upper,
            "lower": lower,
            "bandwidth": (upper - lower) / mid,
        }
    )


def vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Session VWAP over the whole series passed in.

    VWAP = cumsum(typical_price * volume) / cumsum(volume), where
    typical_price = (H + L + C) / 3.

    NOTE: this function does *not* know about session boundaries. Intraday
    VWAP must reset each trading day, so callers pass in a single day's bars
    (which is exactly what the backtest engine does -- strategies are handed a
    per-day slice).
    """
    typical = (high + low + close) / 3.0
    cum_pv = (typical * volume).cumsum()
    cum_v = volume.cumsum()
    return cum_pv / cum_v.replace(0.0, np.nan)


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """Supertrend indicator.

    Basic bands:
        upper_basic = (H + L) / 2 + multiplier * ATR
        lower_basic = (H + L) / 2 - multiplier * ATR

    Final bands ratchet: the upper band can only move down while price stays
    below it, the lower band can only move up while price stays above it.
    Direction flips to long when close crosses above the final upper band and
    to short when it crosses below the final lower band.

    Returns DataFrame with columns: supertrend, direction (+1 long / -1 short),
    upper, lower.
    """
    _check_period(period)
    atr_series = atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    upper_basic = hl2 + multiplier * atr_series
    lower_basic = hl2 - multiplier * atr_series

    n = len(close)
    c = close.to_numpy(dtype=float)
    ub = upper_basic.to_numpy(dtype=float)
    lb = lower_basic.to_numpy(dtype=float)

    final_ub = np.full(n, np.nan)
    final_lb = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    st = np.full(n, np.nan)

    started = False
    for i in range(n):
        if np.isnan(ub[i]) or np.isnan(lb[i]):
            continue
        if not started:
            final_ub[i] = ub[i]
            final_lb[i] = lb[i]
            direction[i] = 1.0 if c[i] > final_ub[i] else -1.0
            st[i] = final_lb[i] if direction[i] > 0 else final_ub[i]
            started = True
            continue

        prev_ub = final_ub[i - 1]
        prev_lb = final_lb[i - 1]
        final_ub[i] = ub[i] if (ub[i] < prev_ub or c[i - 1] > prev_ub) else prev_ub
        final_lb[i] = lb[i] if (lb[i] > prev_lb or c[i - 1] < prev_lb) else prev_lb

        prev_dir = direction[i - 1]
        if prev_dir > 0:
            direction[i] = -1.0 if c[i] < final_lb[i] else 1.0
        else:
            direction[i] = 1.0 if c[i] > final_ub[i] else -1.0
        st[i] = final_lb[i] if direction[i] > 0 else final_ub[i]

    return pd.DataFrame(
        {
            "supertrend": st,
            "direction": direction,
            "upper": final_ub,
            "lower": final_lb,
        },
        index=close.index,
    )


def _check_period(period: int) -> None:
    if not isinstance(period, (int, np.integer)) or period < 1:
        raise ValueError(f"period must be a positive integer, got {period!r}")
