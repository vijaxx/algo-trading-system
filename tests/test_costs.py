"""Indian transaction-cost model arithmetic."""

from __future__ import annotations

import pytest

from algotrade.costs import DEFAULT_COSTS, IndianEquityIntradayCosts


def test_brokerage_uses_flat_floor_on_large_order():
    # turnover = 1000 * 100 = 100,000; 0.03% = 30 > 20 flat -> flat wins
    cost = DEFAULT_COSTS.order_cost(1000.0, 100, "BUY")
    assert cost.brokerage == pytest.approx(20.0)


def test_brokerage_uses_percentage_on_small_order():
    # turnover = 100 * 1 = 100; 0.03% = 0.03 < 20 -> pct wins
    cost = DEFAULT_COSTS.order_cost(100.0, 1, "BUY")
    assert cost.brokerage == pytest.approx(0.03)


def test_stt_only_on_sell_side():
    buy = DEFAULT_COSTS.order_cost(500.0, 100, "BUY")
    sell = DEFAULT_COSTS.order_cost(500.0, 100, "SELL")
    assert buy.stt == pytest.approx(0.0)
    assert sell.stt == pytest.approx(500.0 * 100 * 0.00025)


def test_stamp_duty_only_on_buy_side():
    buy = DEFAULT_COSTS.order_cost(500.0, 100, "BUY")
    sell = DEFAULT_COSTS.order_cost(500.0, 100, "SELL")
    assert buy.stamp_duty == pytest.approx(500.0 * 100 * 0.00003)
    assert sell.stamp_duty == pytest.approx(0.0)


def test_gst_applies_only_to_brokerage_exchange_sebi():
    cost = DEFAULT_COSTS.order_cost(500.0, 100, "SELL")
    expected_gst = 0.18 * (cost.brokerage + cost.exchange_txn + cost.sebi_fees)
    assert cost.gst == pytest.approx(expected_gst)
    # GST must NOT include STT or stamp duty
    not_included = 0.18 * (cost.brokerage + cost.exchange_txn + cost.sebi_fees + cost.stt)
    assert cost.gst != pytest.approx(not_included) or cost.stt == 0.0


def test_total_equals_sum_of_components():
    cost = DEFAULT_COSTS.order_cost(842.35, 37, "SELL")
    total = (
        cost.brokerage
        + cost.stt
        + cost.exchange_txn
        + cost.sebi_fees
        + cost.stamp_duty
        + cost.gst
    )
    assert cost.total == pytest.approx(total)


def test_zero_quantity_zero_cost():
    cost = DEFAULT_COSTS.order_cost(100.0, 0, "BUY")
    assert cost.total == pytest.approx(0.0)


def test_round_trip_long_pays_stt_on_sell_and_stamp_on_buy():
    total = DEFAULT_COSTS.round_trip_cost(100.0, 101.0, 50, direction=1)
    buy_leg = DEFAULT_COSTS.order_cost(100.0, 50, "BUY")
    sell_leg = DEFAULT_COSTS.order_cost(101.0, 50, "SELL")
    assert total == pytest.approx(buy_leg.total + sell_leg.total)


def test_round_trip_short_swaps_sides():
    # short: sell to open, buy to close
    total = DEFAULT_COSTS.round_trip_cost(100.0, 99.0, 50, direction=-1)
    open_leg = DEFAULT_COSTS.order_cost(100.0, 50, "SELL")
    close_leg = DEFAULT_COSTS.order_cost(99.0, 50, "BUY")
    assert total == pytest.approx(open_leg.total + close_leg.total)


def test_breakeven_move_shrinks_with_larger_ticket():
    small = DEFAULT_COSTS.breakeven_move_pct(price=500.0, quantity=10, direction=1)
    large = DEFAULT_COSTS.breakeven_move_pct(price=500.0, quantity=1000, direction=1)
    # small ticket dominated by flat Rs 20 floor -> much higher breakeven %
    assert small > large


def test_invalid_side_raises():
    with pytest.raises(ValueError):
        DEFAULT_COSTS.order_cost(100.0, 10, "HOLD")


def test_invalid_direction_raises():
    with pytest.raises(ValueError):
        DEFAULT_COSTS.round_trip_cost(100.0, 101.0, 10, direction=0)


def test_negative_quantity_raises():
    with pytest.raises(ValueError):
        DEFAULT_COSTS.order_cost(100.0, -5, "BUY")


def test_custom_cost_model_rates_apply():
    custom = IndianEquityIntradayCosts(brokerage_flat=0.0, gst_pct=0.0)
    cost = custom.order_cost(1000.0, 10, "BUY")
    assert cost.brokerage == pytest.approx(0.0)
    assert cost.gst == pytest.approx(0.0)
