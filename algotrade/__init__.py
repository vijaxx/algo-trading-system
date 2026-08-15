"""algotrade -- Indian intraday strategy backtesting and paper-trading engine.

Paper/simulation only. This package contains no live order-routing code.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .costs import DEFAULT_COSTS, IndianEquityIntradayCosts
from .engine import BacktestEngine, BacktestResult
from .metrics import PerformanceReport, Trade, build_report
from .risk import RiskConfig, RiskManager
from .strategies import STRATEGY_REGISTRY, build_strategy

__all__ = [
    "__version__",
    "BacktestEngine",
    "BacktestResult",
    "IndianEquityIntradayCosts",
    "DEFAULT_COSTS",
    "PerformanceReport",
    "Trade",
    "build_report",
    "RiskConfig",
    "RiskManager",
    "STRATEGY_REGISTRY",
    "build_strategy",
]
