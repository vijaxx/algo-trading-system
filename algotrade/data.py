"""Deterministic synthetic NSE intraday OHLCV data.

Why synthetic: the repo must run end-to-end straight after `git clone`, with
no API key, no vendor account and no network call. Real NSE tick/minute data
is licensed and cannot be redistributed. So we generate bars that have the
*structural* features intraday strategies react to:

  * overnight gaps (open != previous close),
  * a U-shaped intraday volatility curve (busy open, quiet lunch, busy close),
  * a matching U-shaped volume profile,
  * mild intraday trend/mean-reversion mixture so that both breakout and
    reversion strategies have something to find,
  * per-symbol price level, volatility and drift.

Everything is seeded, so two runs on two machines produce byte-identical bars.
The generated numbers are NOT real market data and results from them say
nothing about live performance -- see the README limitations section.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

# NSE equity cash session
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
# Regulator/broker convention: intraday (MIS) positions are auto-squared-off
# by the broker shortly before the close. 15:15 is the common cutoff.
SQUARE_OFF_TIME = time(15, 15)


@dataclass(frozen=True)
class SymbolSpec:
    """Static description of a synthetic instrument."""

    symbol: str
    start_price: float
    daily_vol: float  # stdev of daily return, e.g. 0.015 = 1.5%
    gap_vol: float  # stdev of the overnight gap
    base_volume: int  # average volume in a mid-session 5-minute bar
    trendiness: float  # 0 = pure noise, 1 = strongly autocorrelated
    tick_size: float = 0.05


# A handful of liquid NSE names, with plausible-but-invented characteristics.
DEFAULT_UNIVERSE: tuple = (
    SymbolSpec("RELIANCE", 2850.0, 0.013, 0.006, 42000, 0.55),
    SymbolSpec("TCS", 3900.0, 0.011, 0.005, 18000, 0.45),
    SymbolSpec("HDFCBANK", 1620.0, 0.012, 0.005, 55000, 0.35),
    SymbolSpec("INFY", 1480.0, 0.014, 0.007, 36000, 0.50),
    SymbolSpec("SBIN", 790.0, 0.018, 0.008, 90000, 0.60),
    SymbolSpec("TATAMOTORS", 985.0, 0.022, 0.010, 75000, 0.65),
)

SYMBOLS = tuple(s.symbol for s in DEFAULT_UNIVERSE)
_SPEC_BY_SYMBOL = {s.symbol: s for s in DEFAULT_UNIVERSE}


def trading_days(start: date, count: int) -> list:
    """`count` weekdays starting at (or after) `start`.

    Exchange holidays are not modelled -- for synthetic data it makes no
    difference, and hard-coding an NSE holiday list would go stale.
    """
    days = []
    day = start
    while len(days) < count:
        if day.weekday() < 5:  # Mon-Fri
            days.append(day)
        day += timedelta(days=1)
    return days


def session_timestamps(day: date, bar_minutes: int) -> list:
    """Bar timestamps for one session.

    A bar is labelled by its OPEN time, so the 09:15 bar covers 09:15-09:20
    for 5-minute bars. The last bar starts strictly before 15:30.
    """
    if 375 % bar_minutes != 0:
        # 09:15 -> 15:30 is 375 minutes.
        raise ValueError(
            f"bar_minutes={bar_minutes} does not divide the 375-minute NSE session"
        )
    start = datetime.combine(day, MARKET_OPEN)
    n_bars = 375 // bar_minutes
    return [start + timedelta(minutes=bar_minutes * i) for i in range(n_bars)]


def _intraday_shape(n_bars: int) -> np.ndarray:
    """U-shaped multiplier over the session, ~1.8x at the edges, ~0.6x midday.

    Real intraday volatility and volume both follow this smile; strategies
    that size stops off ATR behave very differently with and without it.
    """
    x = np.linspace(0.0, 1.0, n_bars)
    return 0.6 + 1.2 * (2.0 * x - 1.0) ** 2


def generate_symbol(
    spec: SymbolSpec,
    days: list,
    bar_minutes: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic OHLCV bars for one symbol.

    The seed is combined with a stable hash of the symbol name so each symbol
    gets its own independent path while the whole dataset stays reproducible.
    """
    symbol_seed = (seed * 1000003 + _stable_hash(spec.symbol)) % (2**32)
    rng = np.random.default_rng(symbol_seed)

    n_bars = 375 // bar_minutes
    shape = _intraday_shape(n_bars)
    # Convert a daily return stdev into a per-bar stdev (sqrt of time).
    bar_vol = spec.daily_vol / np.sqrt(n_bars)

    rows = []
    prev_close = spec.start_price

    for day in days:
        gap = rng.normal(0.0, spec.gap_vol)
        day_open = prev_close * (1.0 + gap)

        # Per-day drift: small, so days are not all trending the same way.
        drift = rng.normal(0.0, spec.daily_vol * 0.35) / n_bars

        price = day_open
        prev_shock = 0.0
        timestamps = session_timestamps(day, bar_minutes)

        for i, ts in enumerate(timestamps):
            vol_i = bar_vol * shape[i]
            # AR(1) shock: trendiness controls how much of the previous move
            # carries into this bar. High trendiness -> momentum-friendly,
            # low -> mean-reversion-friendly.
            eps = rng.normal(0.0, 1.0)
            shock = spec.trendiness * prev_shock + np.sqrt(
                max(1.0 - spec.trendiness**2, 1e-9)
            ) * eps
            prev_shock = shock
            ret = drift + vol_i * shock

            bar_open = price
            bar_close = bar_open * (1.0 + ret)

            # Wick sizes scale with the bar's own realised range.
            body = abs(bar_close - bar_open)
            wick_scale = vol_i * bar_open
            up_wick = abs(rng.normal(0.0, 0.7)) * wick_scale
            dn_wick = abs(rng.normal(0.0, 0.7)) * wick_scale
            bar_high = max(bar_open, bar_close) + up_wick + 0.15 * body
            bar_low = min(bar_open, bar_close) - dn_wick - 0.15 * body
            bar_low = max(bar_low, 0.01)

            # Volume: U-shape * lognormal noise * a kick when the bar moves.
            move_factor = 1.0 + 3.0 * abs(ret) / max(bar_vol, 1e-9) * 0.15
            vol_noise = float(rng.lognormal(0.0, 0.28))
            volume = int(spec.base_volume * shape[i] * vol_noise * move_factor)

            rows.append(
                {
                    "timestamp": ts,
                    "symbol": spec.symbol,
                    "open": _round_tick(bar_open, spec.tick_size),
                    "high": _round_tick(bar_high, spec.tick_size),
                    "low": _round_tick(bar_low, spec.tick_size),
                    "close": _round_tick(bar_close, spec.tick_size),
                    "volume": max(volume, 1),
                }
            )
            price = bar_close

        prev_close = price

    df = pd.DataFrame(rows)
    df = df.set_index("timestamp").sort_index()
    # Tick rounding can push close outside [low, high] by half a tick; repair
    # so the OHLC invariant high >= max(o,c) >= min(o,c) >= low always holds.
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def generate_dataset(
    symbols=None,
    n_days: int = 60,
    start: date = date(2024, 1, 1),
    bar_minutes: int = 5,
    seed: int = 42,
) -> dict:
    """Generate `{symbol: DataFrame}` for the requested universe."""
    if symbols is None:
        specs = list(DEFAULT_UNIVERSE)
    else:
        specs = []
        for s in symbols:
            if s not in _SPEC_BY_SYMBOL:
                raise KeyError(
                    f"unknown symbol {s!r}; known symbols: {', '.join(SYMBOLS)}"
                )
            specs.append(_SPEC_BY_SYMBOL[s])

    days = trading_days(start, n_days)
    return {
        spec.symbol: generate_symbol(spec, days, bar_minutes=bar_minutes, seed=seed)
        for spec in specs
    }


def _round_tick(price: float, tick: float) -> float:
    return round(round(price / tick) * tick, 2)


def _stable_hash(text: str) -> int:
    """Deterministic across processes (unlike builtin hash() for str)."""
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) % (2**32)
    return h
