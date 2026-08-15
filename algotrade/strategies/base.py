"""Shared strategy interface.

The single most important contract in this file is the one on
`generate_signal`: a strategy is handed **only** the bars of the current
session up to and including the bar that just closed. It is structurally
unable to see the future, because the future rows are not in the DataFrame it
receives. See engine.py and tests/test_no_lookahead.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

LONG = 1
SHORT = -1


@dataclass(frozen=True)
class Signal:
    """An entry instruction produced at the close of a bar.

    The engine fills it at the NEXT bar's open (plus slippage), never at the
    close of the bar that produced it -- filling at the signal bar's own close
    would be a subtle form of lookahead, since in live trading the close is
    only known once the bar is already gone.
    """

    side: int  # LONG (+1) or SHORT (-1)
    stop_loss: float
    target: float
    reason: str = ""

    def __post_init__(self) -> None:
        if self.side not in (LONG, SHORT):
            raise ValueError(f"side must be +1 or -1, got {self.side}")
        if self.stop_loss <= 0 or self.target <= 0:
            raise ValueError("stop_loss and target must be positive prices")


class Strategy(ABC):
    """Base class for every intraday strategy in this package."""

    #: human readable, also the CLI key
    name: str = "base"

    def __init__(self, **params) -> None:
        self.params = params

    # ---- lifecycle ------------------------------------------------------
    def on_session_start(self, day: date) -> None:
        """Reset any per-day state. Called once before each trading session.

        Intraday strategies must not carry state across days -- opening
        ranges, session VWAP and daily counters all reset at 09:15.
        """

    # ---- signal generation ---------------------------------------------
    @abstractmethod
    def generate_signal(self, history: pd.DataFrame) -> Optional[Signal]:
        """Return an entry Signal, or None.

        `history` contains the current session's bars from 09:15 up to and
        including the bar that has just closed. `history.iloc[-1]` is the most
        recent completed bar. Rows after it do not exist.

        Only called when the engine is flat and risk checks have passed.
        """

    def should_exit(self, history: pd.DataFrame, side: int) -> bool:
        """Optional discretionary exit, checked at each bar close while in a
        position. Stop-loss/target/square-off are handled by the engine and
        are always active regardless of what this returns.
        """
        return False

    # ---- helpers --------------------------------------------------------
    def __repr__(self) -> str:
        joined = ", ".join(f"{k}={v!r}" for k, v in sorted(self.params.items()))
        return f"{self.__class__.__name__}({joined})"

    @property
    def description(self) -> str:
        return (self.__doc__ or "").strip().splitlines()[0] if self.__doc__ else ""
