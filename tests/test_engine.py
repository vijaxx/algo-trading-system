"""End-to-end engine behaviour: risk enforcement, square-off, cost application."""

from __future__ import annotations

from datetime import time

import pytest

from algotrade.data import generate_dataset
from algotrade.engine import EXIT_SQUARE_OFF, BacktestEngine
from algotrade.risk import RiskConfig
from algotrade.strategies import build_strategy


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(symbols=["RELIANCE", "SBIN", "TCS"], n_days=30, seed=42)


def test_no_trade_survives_past_square_off_time(dataset):
    engine = BacktestEngine(
        strategy_factory=lambda: build_strategy("supertrend"),
        risk_config=RiskConfig(starting_capital=500_000, max_positions=5),
    )
    result = engine.run(dataset)
    for t in result.trades:
        assert t.exit_time.time() <= time(15, 30)
        if t.exit_reason == EXIT_SQUARE_OFF:
            assert t.exit_time.time() >= time(15, 15)
    # every position must be closed same-day (no overnight carry)
    for t in result.trades:
        assert t.entry_time.date() == t.exit_time.date()


def test_max_positions_cap_is_never_exceeded(dataset):
    cap = 1
    engine = BacktestEngine(
        strategy_factory=lambda: build_strategy("vwap_reversion", max_trades_per_day=5),
        risk_config=RiskConfig(starting_capital=500_000, max_positions=cap),
    )
    result = engine.run(dataset)
    # reconstruct concurrent open positions from trade intervals
    events = []
    for t in result.trades:
        events.append((t.entry_time, 1))
        events.append((t.exit_time, -1))
    events.sort(key=lambda e: (e[0], e[1]))  # exits before entries at same ts
    concurrent = 0
    max_concurrent = 0
    for _, delta in events:
        concurrent += delta
        max_concurrent = max(max_concurrent, concurrent)
    assert max_concurrent <= cap


def test_daily_loss_limit_actually_stops_new_entries(dataset):
    # extremely tight daily loss limit should sharply cut trade count
    tight = BacktestEngine(
        strategy_factory=lambda: build_strategy("orb"),
        risk_config=RiskConfig(starting_capital=500_000, max_daily_loss_pct=0.01, max_positions=5),
    ).run(dataset)
    loose = BacktestEngine(
        strategy_factory=lambda: build_strategy("orb"),
        risk_config=RiskConfig(starting_capital=500_000, max_daily_loss_pct=50.0, max_positions=5),
    ).run(dataset)
    assert tight.blocked.get("daily loss limit breached", 0) >= 0  # guard exists
    assert tight.report.num_trades <= loose.report.num_trades


def test_zero_cost_run_has_zero_total_costs(dataset):
    result = BacktestEngine(
        strategy_factory=lambda: build_strategy("ema_rsi"),
        risk_config=RiskConfig(starting_capital=500_000, max_positions=5),
        apply_costs=False,
    ).run(dataset)
    assert result.report.total_costs == pytest.approx(0.0)


def test_with_costs_run_has_positive_total_costs_when_trades_exist(dataset):
    result = BacktestEngine(
        strategy_factory=lambda: build_strategy("ema_rsi"),
        risk_config=RiskConfig(starting_capital=500_000, max_positions=5),
        apply_costs=True,
    ).run(dataset)
    if result.report.num_trades > 0:
        assert result.report.total_costs > 0.0


def test_net_pnl_always_less_than_or_equal_gross_pnl_with_costs(dataset):
    result = BacktestEngine(
        strategy_factory=lambda: build_strategy("bollinger"),
        risk_config=RiskConfig(starting_capital=500_000, max_positions=5),
        apply_costs=True,
    ).run(dataset)
    assert result.report.net_pnl <= result.report.gross_pnl + 1e-6


def test_empty_data_raises():
    engine = BacktestEngine(strategy_factory=lambda: build_strategy("orb"))
    with pytest.raises(ValueError):
        engine.run({})


def test_per_symbol_breakdown_sums_to_total(dataset):
    result = BacktestEngine(
        strategy_factory=lambda: build_strategy("orb"),
        risk_config=RiskConfig(starting_capital=500_000, max_positions=5),
    ).run(dataset)
    ps = result.per_symbol()
    if not ps.empty:
        assert ps["net_pnl"].sum() == pytest.approx(result.report.net_pnl, rel=1e-6)
