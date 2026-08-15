"""Performance metrics math."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from algotrade.metrics import Trade, build_report, max_drawdown, sharpe_ratio


def make_trade(net_gross, costs, day, symbol="RELIANCE"):
    return Trade(
        symbol=symbol,
        strategy="test",
        side=1,
        quantity=10,
        entry_time=pd.Timestamp(f"{day} 10:00"),
        entry_price=100.0,
        exit_time=pd.Timestamp(f"{day} 10:30"),
        exit_price=100.0 + net_gross / 10.0,
        exit_reason="target",
        gross_pnl=net_gross,
        costs=costs,
    )


def test_win_rate_and_counts():
    trades = [
        make_trade(500, 20, "2024-01-01"),
        make_trade(-200, 20, "2024-01-01"),
        make_trade(300, 20, "2024-01-02"),
    ]
    r = build_report(trades, starting_capital=100_000)
    assert r.num_trades == 3
    assert r.wins == 2
    assert r.losses == 1
    assert r.win_rate_pct == pytest.approx(200 / 3)


def test_net_pnl_equals_gross_minus_costs():
    trades = [make_trade(1000, 50, "2024-01-01"), make_trade(-400, 50, "2024-01-02")]
    r = build_report(trades, starting_capital=100_000)
    assert r.gross_pnl == pytest.approx(600)
    assert r.total_costs == pytest.approx(100)
    assert r.net_pnl == pytest.approx(500)


def test_profit_factor_hand_computed():
    # wins net: 480, 280 = 760 ; losses net: -220 = 220
    trades = [
        make_trade(500, 20, "2024-01-01"),
        make_trade(300, 20, "2024-01-02"),
        make_trade(-200, 20, "2024-01-03"),
    ]
    r = build_report(trades, starting_capital=100_000)
    assert r.profit_factor == pytest.approx(760 / 220)


def test_profit_factor_infinite_when_no_losses():
    trades = [make_trade(100, 10, "2024-01-01")]
    r = build_report(trades, starting_capital=100_000)
    assert math.isinf(r.profit_factor)


def test_profit_factor_zero_when_no_trades():
    r = build_report([], starting_capital=100_000)
    assert r.profit_factor == 0.0
    assert r.num_trades == 0


def test_expectancy_is_net_pnl_over_trades():
    trades = [make_trade(500, 20, "2024-01-01"), make_trade(-300, 20, "2024-01-02")]
    r = build_report(trades, starting_capital=100_000)
    assert r.expectancy == pytest.approx(r.net_pnl / 2)


def test_max_drawdown_simple_curve():
    curve = pd.Series([100.0, 120.0, 90.0, 95.0, 130.0])
    dd_abs, dd_pct = max_drawdown(curve)
    assert dd_abs == pytest.approx(30.0)  # 120 -> 90
    assert dd_pct == pytest.approx(25.0)  # 30/120


def test_max_drawdown_zero_for_monotonic_up():
    curve = pd.Series([100.0, 110.0, 120.0, 130.0])
    dd_abs, dd_pct = max_drawdown(curve)
    assert dd_abs == pytest.approx(0.0)
    assert dd_pct == pytest.approx(0.0)


def test_sharpe_zero_when_flat_returns():
    returns = pd.Series([0.001, 0.001, 0.001, 0.001])
    assert sharpe_ratio(returns) == pytest.approx(0.0)


def test_sharpe_positive_for_consistently_positive_returns():
    returns = pd.Series([0.01, 0.02, 0.005, 0.015, 0.008])
    s = sharpe_ratio(returns)
    assert s > 0


def test_sharpe_zero_for_single_point():
    assert sharpe_ratio(pd.Series([0.01])) == 0.0


def test_return_pct_relative_to_starting_capital():
    trades = [make_trade(5000, 200, "2024-01-01")]
    r = build_report(trades, starting_capital=200_000)
    assert r.return_pct == pytest.approx(100.0 * 4800 / 200_000)


def test_cost_drag_pct_of_gross():
    trades = [make_trade(1000, 100, "2024-01-01")]
    r = build_report(trades, starting_capital=100_000)
    assert r.cost_drag_pct_of_gross == pytest.approx(10.0)
