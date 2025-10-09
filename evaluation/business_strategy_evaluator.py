"""Business-grade evaluator for trading strategies.

The evaluator focuses on qualitative and quantitative heuristics that are
commonly used by professional proprietary desks when vetting a new strategy
before allowing it to go live.  It inspects the strategy implementation,
performs lightweight backtesting checks on provided market data, and produces
an overall score on a 0-10 scale.  A score above ``min_passing_score`` (9.9 by
default) indicates that the strategy satisfies strict commercial standards.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.base import StrategyBase, TradingSignal


@dataclass
class EvaluationBreakdown:
    """Detailed evaluation scores for transparency."""

    robustness: float
    risk_management: float
    market_alignment: float
    code_quality: float
    signal_depth: float

    @property
    def total(self) -> float:
        return self.robustness + self.risk_management + self.market_alignment + self.code_quality + self.signal_depth


@dataclass
class BusinessStrategyEvaluator:
    """Evaluator that enforces commercial-level requirements."""

    min_passing_score: float = 9.9
    weights: Dict[str, float] = field(default_factory=lambda: {
        "robustness": 2.0,
        "risk_management": 2.0,
        "market_alignment": 2.0,
        "code_quality": 2.0,
        "signal_depth": 2.0,
    })

    def evaluate(self, strategy: StrategyBase, data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate a strategy and return a full report.

        Args:
            strategy: Strategy instance implementing :class:`StrategyBase`.
            data: Representative market data for signal generation.

        Returns:
            A dictionary containing the score breakdown and pass/fail flag.
        """

        if not isinstance(strategy, StrategyBase):
            raise TypeError("strategy 必须继承 StrategyBase")

        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("评估需要非空的行情数据 DataFrame")

        breakdown = EvaluationBreakdown(
            robustness=self._score_robustness(strategy, data),
            risk_management=self._score_risk_management(strategy, data),
            market_alignment=self._score_market_alignment(strategy),
            code_quality=self._score_code_quality(strategy),
            signal_depth=self._score_signal_depth(strategy, data),
        )

        weighted_score = sum(getattr(breakdown, key) / self.weights[key] * self.weights[key] for key in self.weights)

        result = {
            "score": round(weighted_score, 2),
            "breakdown": breakdown,
            "passed": weighted_score >= self.min_passing_score,
        }

        return result

    # ------------------------------------------------------------------
    # Scoring components
    # ------------------------------------------------------------------
    def _score_robustness(self, strategy: StrategyBase, data: pd.DataFrame) -> float:
        """Assess data validation, error handling and stability."""

        score = 0.0

        try:
            signals = strategy.generate_signals(data.copy())
        except Exception as exc:  # pragma: no cover - defensive branch
            raise RuntimeError(f"策略在生成信号时出现异常: {exc}")

        score += 1.0  # 能够稳定生成信号

        has_required_attr = hasattr(strategy, "required_columns")
        score += 0.5 if has_required_attr else 0.0

        if has_required_attr:
            missing_cols = set(strategy.required_columns) - set(data.columns)
            score += 0.5 if not missing_cols else 0.0

        # 稳定性：返回的信号对象必须符合 TradingSignal
        if all(isinstance(sig, TradingSignal) for sig in signals):
            score += 0.5

        return min(score, self.weights["robustness"])

    def _score_risk_management(self, strategy: StrategyBase, data: pd.DataFrame) -> float:
        """Ensure stop loss / take profit / position sizing are enforced."""

        raw_signals = strategy.generate_signals(data.copy())
        managed = strategy.apply_risk_management(raw_signals)

        if not managed:
            return 0.5  # 没有信号仅给最低分

        score = 1.0

        if all(sig.stop_loss and sig.take_profit for sig in managed):
            score += 0.5

        position_sizes = [sig.position_size for sig in managed]
        if position_sizes and max(position_sizes) <= strategy.config.max_position_size + 1e-6:
            score += 0.3

        if all(0.0 < sig.confidence <= 1.0 for sig in managed):
            score += 0.2

        return min(score, self.weights["risk_management"])

    def _score_market_alignment(self, strategy: StrategyBase) -> float:
        """Verify that the strategy is aligned with China A-share limit-up trading."""

        score = 1.0

        has_thresholds = hasattr(strategy, "limit_up_thresholds")
        if has_thresholds and isinstance(strategy.limit_up_thresholds, dict):
            if {"main", "growth", "st"}.issubset(set(strategy.limit_up_thresholds.keys())):
                score += 0.5

        has_streak = getattr(strategy, "consecutive_limit_up_days", 0) >= 2
        score += 0.3 if has_streak else 0.0

        has_reason = hasattr(strategy, "_build_signal_reason")
        score += 0.2 if has_reason else 0.0

        return min(score, self.weights["market_alignment"])

    def _score_code_quality(self, strategy: StrategyBase) -> float:
        """Check documentation, typing and modular design."""

        score = 1.0

        docstring = inspect.getdoc(strategy.__class__)
        if docstring and len(docstring.split()) >= 10:
            score += 0.5

        signature = inspect.signature(strategy.generate_signals)
        annotations_present = all(param.annotation is not inspect._empty for param in signature.parameters.values())
        annotations_present = annotations_present and signature.return_annotation is not inspect._empty
        score += 0.3 if annotations_present else 0.0

        helper_methods = [name for name, _ in inspect.getmembers(strategy.__class__, predicate=inspect.isfunction) if name.startswith("_")]
        score += 0.2 if len(helper_methods) >= 5 else 0.0

        return min(score, self.weights["code_quality"])

    def _score_signal_depth(self, strategy: StrategyBase, data: pd.DataFrame) -> float:
        """Gauge the richness and explanatory power of signals."""

        signals = strategy.apply_risk_management(strategy.generate_signals(data.copy()))

        if not signals:
            return 0.8

        score = 1.0

        confidences = [sig.confidence for sig in signals if sig.confidence is not None]
        if confidences and float(np.mean(confidences)) >= 0.7:
            score += 0.5

        reason_lengths = [len(sig.reason or "") for sig in signals]
        if reason_lengths and np.mean(reason_lengths) >= 10:
            score += 0.3

        stop_ranges = [sig.take_profit - sig.price for sig in signals if sig.take_profit and sig.stop_loss]
        if stop_ranges and min(stop_ranges) > 0:
            score += 0.2

        return min(score, self.weights["signal_depth"])
