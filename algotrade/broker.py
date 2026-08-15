"""Broker abstraction.

**This package cannot place a real order.** `BrokerAdapter` is an interface,
`PaperBroker` is a full simulation, and `SketchLiveBrokerAdapter` is a
deliberately unfinished outline whose order methods raise
`NotImplementedError`. There is no HTTP client, no credential handling and no
order-placement code path anywhere in this repository. See the README.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .costs import DEFAULT_COSTS, CostBreakdown, IndianEquityIntradayCosts

BUY = "BUY"
SELL = "SELL"


@dataclass(frozen=True)
class Order:
    """A market order request."""

    symbol: str
    side: str  # BUY / SELL
    quantity: int
    timestamp: datetime
    order_type: str = "MARKET"
    product: str = "MIS"  # intraday margin product

    def __post_init__(self) -> None:
        if self.side not in (BUY, SELL):
            raise ValueError(f"side must be BUY or SELL, got {self.side!r}")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")


@dataclass(frozen=True)
class Fill:
    """An executed order."""

    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    timestamp: datetime
    charges: CostBreakdown

    @property
    def value(self) -> float:
        return self.price * self.quantity


class BrokerAdapter(ABC):
    """Interface every broker implementation must satisfy."""

    @abstractmethod
    def connect(self) -> None:
        """Establish a session. Paper broker is a no-op."""

    @abstractmethod
    def place_order(self, order: Order, reference_price: float) -> Fill:
        """Submit an order and return the resulting fill."""

    @abstractmethod
    def get_positions(self) -> Dict[str, int]:
        """Net quantity per symbol (positive long, negative short)."""

    @abstractmethod
    def get_funds(self) -> float:
        """Available cash/equity."""


class PaperBroker(BrokerAdapter):
    """Fully working simulated broker.

    Fill model: market orders fill at `reference_price` adjusted by
    `slippage_bps` in the direction that hurts (buys fill higher, sells fill
    lower). Charges come from the Indian intraday cost model.

    This is intentionally simple and intentionally pessimistic-by-default. It
    does NOT model queue position, partial fills, or impact -- see the README
    limitations section.
    """

    def __init__(
        self,
        starting_funds: float = 500_000.0,
        slippage_bps: float = 2.0,
        cost_model: Optional[IndianEquityIntradayCosts] = None,
    ) -> None:
        self.starting_funds = starting_funds
        self.funds = starting_funds
        self.slippage_bps = slippage_bps
        self.cost_model = cost_model or DEFAULT_COSTS
        self.positions: Dict[str, int] = {}
        self.fills: List[Fill] = []
        self._ids = itertools.count(1)
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def fill_price(self, reference_price: float, side: str) -> float:
        """Apply slippage. 1 bp = 0.01%."""
        adj = reference_price * self.slippage_bps / 10_000.0
        return reference_price + adj if side == BUY else reference_price - adj

    def place_order(self, order: Order, reference_price: float) -> Fill:
        if reference_price <= 0:
            raise ValueError("reference_price must be positive")
        price = self.fill_price(reference_price, order.side)
        charges = self.cost_model.order_cost(price, order.quantity, order.side)

        signed = order.quantity if order.side == BUY else -order.quantity
        self.positions[order.symbol] = self.positions.get(order.symbol, 0) + signed
        if self.positions[order.symbol] == 0:
            del self.positions[order.symbol]

        # Cash effect: pay for buys, receive on sells, always pay charges.
        self.funds += (-price * order.quantity if order.side == BUY else price * order.quantity)
        self.funds -= charges.total

        fill = Fill(
            order_id=f"PAPER-{next(self._ids):06d}",
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=price,
            timestamp=order.timestamp,
            charges=charges,
        )
        self.fills.append(fill)
        return fill

    def get_positions(self) -> Dict[str, int]:
        return dict(self.positions)

    def get_funds(self) -> float:
        return self.funds


class SketchLiveBrokerAdapter(BrokerAdapter):
    """UNFINISHED OUTLINE -- cannot and must not place live orders.

    Kept in the repo only to show where a real adapter (Angel One SmartAPI,
    Fyers API v3, Upstox API v2) would slot into the `BrokerAdapter`
    interface. Every order-facing method raises `NotImplementedError`. There
    is no HTTP client, no auth flow, no API key handling and no endpoint URL
    in this class, by design.

    Anyone completing this would additionally need: a SEBI-registered broker
    account, exchange approval for algorithmic order flow (the broker's algo
    ID must be tagged on every order), and their own testing on the broker's
    sandbox. Do not point this at a funded account.
    """

    def __init__(self, broker_name: str = "angelone") -> None:
        self.broker_name = broker_name

    def connect(self) -> None:
        raise NotImplementedError(
            f"{self.broker_name} live connectivity is intentionally not implemented; "
            "use PaperBroker"
        )

    def place_order(self, order: Order, reference_price: float) -> Fill:
        raise NotImplementedError(
            "live order placement is intentionally not implemented in this project"
        )

    def get_positions(self) -> Dict[str, int]:
        raise NotImplementedError("live position fetch is not implemented")

    def get_funds(self) -> float:
        raise NotImplementedError("live funds fetch is not implemented")
