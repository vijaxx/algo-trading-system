"""Risk / SEBI-style compliance guards must actually block trades."""

from __future__ import annotations

from datetime import time

import pytest

from algotrade.risk import RiskConfig, RiskManager


def make_manager(**overrides) -> RiskManager:
    defaults = dict(
        starting_capital=100_000.0,
        max_daily_loss_pct=2.0,
        max_positions=2,
        capital_per_trade_pct=20.0,
        max_risk_per_trade_pct=1.0,
        circuit_limit_pct=10.0,
    )
    defaults.update(overrides)
    cfg = RiskConfig(**defaults)
    return RiskManager(cfg)


def test_daily_loss_limit_blocks_further_entries():
    rm = make_manager()
    rm.start_day()
    assert rm.check_daily_loss().allowed
    rm.record_trade(-2500.0)  # exactly 2.5% loss > 2% limit
    decision = rm.check_daily_loss()
    assert not decision.allowed
    assert "daily loss" in decision.reason

    composite = rm.can_enter(time(10, 0), 100.0, 100.0)
    assert not composite.allowed


def test_daily_loss_limit_does_not_block_before_breach():
    rm = make_manager()
    rm.start_day()
    rm.record_trade(-500.0)  # 0.5% loss, within 2% limit
    assert rm.check_daily_loss().allowed


def test_max_positions_blocks_extra_entries():
    rm = make_manager(max_positions=1)
    rm.open_positions = 1
    decision = rm.check_position_count()
    assert not decision.allowed
    assert "max positions" in decision.reason


def test_max_positions_allows_under_the_cap():
    rm = make_manager(max_positions=2)
    rm.open_positions = 1
    assert rm.check_position_count().allowed


def test_square_off_time_blocks_new_entries():
    rm = make_manager()
    decision = rm.check_entry_time(time(15, 20))
    assert not decision.allowed


def test_no_entry_after_cutoff_blocks_before_square_off():
    rm = make_manager()
    # default no_entry_after=15:00, square_off=15:15
    decision = rm.check_entry_time(time(15, 5))
    assert not decision.allowed
    assert "cutoff" in decision.reason


def test_entry_allowed_during_normal_hours():
    rm = make_manager()
    assert rm.check_entry_time(time(11, 0)).allowed


def test_must_square_off_flag():
    rm = make_manager()
    assert rm.must_square_off(time(15, 15)) is True
    assert rm.must_square_off(time(15, 14)) is False


def test_circuit_breaker_blocks_extreme_move():
    rm = make_manager(circuit_limit_pct=5.0)
    decision = rm.check_circuit_breaker(price=110.0, reference_price=100.0)
    assert not decision.allowed
    assert "circuit breaker" in decision.reason


def test_circuit_breaker_allows_normal_move():
    rm = make_manager(circuit_limit_pct=5.0)
    decision = rm.check_circuit_breaker(price=102.0, reference_price=100.0)
    assert decision.allowed


def test_can_enter_composite_blocks_on_first_failing_guard():
    rm = make_manager(max_positions=1)
    rm.open_positions = 1
    decision = rm.can_enter(time(11, 0), 100.0, 100.0)
    assert not decision.allowed
    assert "positions" in decision.reason


def test_position_sizing_respects_capital_cap():
    rm = make_manager(capital_per_trade_pct=10.0, max_risk_per_trade_pct=100.0)
    # capital cap: 10% of 100,000 = 10,000 / price 100 = 100 shares
    # risk cap: essentially unlimited (100% risk budget, huge stop distance)
    qty = rm.position_size(price=100.0, stop_loss=99.99)
    assert qty <= 100


def test_position_sizing_respects_risk_cap():
    rm = make_manager(capital_per_trade_pct=100.0, max_risk_per_trade_pct=1.0)
    # risk budget = 1% of 100,000 = 1000; stop distance = 10 -> qty = 100
    qty = rm.position_size(price=100.0, stop_loss=90.0)
    assert qty == 100


def test_position_sizing_zero_when_stop_equals_price():
    rm = make_manager()
    qty = rm.position_size(price=100.0, stop_loss=100.0)
    assert qty == 0


def test_blocked_counts_tracked():
    rm = make_manager(max_positions=0 + 1)
    rm.open_positions = 1
    rm.can_enter(time(11, 0), 100.0, 100.0)
    rm.can_enter(time(11, 5), 100.0, 100.0)
    assert sum(rm.blocked_counts.values()) == 2


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        RiskConfig(starting_capital=-1)
    with pytest.raises(ValueError):
        RiskConfig(max_positions=0)
