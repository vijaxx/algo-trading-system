"""PaperBroker fills, charges and the unfinished live-adapter guard."""

from __future__ import annotations

from datetime import datetime

import pytest

from algotrade.broker import BUY, SELL, Order, PaperBroker, SketchLiveBrokerAdapter


def test_buy_fills_above_reference_with_slippage():
    b = PaperBroker(starting_funds=100_000, slippage_bps=10.0)
    b.connect()
    fill = b.place_order(
        Order("RELIANCE", BUY, 10, datetime(2024, 1, 1, 9, 20)), reference_price=1000.0
    )
    assert fill.price > 1000.0
    assert fill.price == pytest.approx(1001.0)  # 10 bps = 0.1%


def test_sell_fills_below_reference_with_slippage():
    b = PaperBroker(starting_funds=100_000, slippage_bps=10.0)
    fill = b.place_order(
        Order("RELIANCE", SELL, 10, datetime(2024, 1, 1, 9, 20)), reference_price=1000.0
    )
    assert fill.price < 1000.0


def test_funds_decrease_on_buy_by_price_plus_charges():
    b = PaperBroker(starting_funds=100_000, slippage_bps=0.0)
    fill = b.place_order(
        Order("TCS", BUY, 5, datetime(2024, 1, 1, 9, 20)), reference_price=2000.0
    )
    expected = 100_000 - 2000.0 * 5 - fill.charges.total
    assert b.get_funds() == pytest.approx(expected)


def test_positions_tracked_and_netted_to_zero():
    b = PaperBroker(starting_funds=100_000, slippage_bps=0.0)
    b.place_order(Order("TCS", BUY, 5, datetime(2024, 1, 1, 9, 20)), reference_price=2000.0)
    assert b.get_positions()["TCS"] == 5
    b.place_order(Order("TCS", SELL, 5, datetime(2024, 1, 1, 9, 25)), reference_price=2010.0)
    assert "TCS" not in b.get_positions()


def test_order_rejects_zero_quantity():
    with pytest.raises(ValueError):
        Order("TCS", BUY, 0, datetime(2024, 1, 1, 9, 20))


def test_order_rejects_bad_side():
    with pytest.raises(ValueError):
        Order("TCS", "HOLD", 10, datetime(2024, 1, 1, 9, 20))


def test_sketch_live_adapter_cannot_connect_or_trade():
    adapter = SketchLiveBrokerAdapter("angelone")
    with pytest.raises(NotImplementedError):
        adapter.connect()
    with pytest.raises(NotImplementedError):
        adapter.place_order(
            Order("TCS", BUY, 1, datetime(2024, 1, 1, 9, 20)), reference_price=100.0
        )
    with pytest.raises(NotImplementedError):
        adapter.get_positions()
    with pytest.raises(NotImplementedError):
        adapter.get_funds()
