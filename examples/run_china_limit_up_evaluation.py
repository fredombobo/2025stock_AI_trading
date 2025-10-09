"""Example script to evaluate the China limit-up strategy."""

from __future__ import annotations

import os
import sys
import types

import pandas as pd

# Ensure external dependencies that require environment variables are satisfied
os.environ.setdefault("TUSHARE_TOKEN", "demo-token-for-testing")

# Inject lightweight stubs for heavy infrastructure modules used by StrategyBase.
if "data" not in sys.modules:
    data_module = types.ModuleType("data")
    processors_module = types.ModuleType("data.processors")
    storage_module = types.ModuleType("data.storage")
    database_manager_module = types.ModuleType("data.storage.database_manager")

    dummy_repo = types.SimpleNamespace(
        get_daily_price=lambda *args, **kwargs: pd.DataFrame(),
        get_technical_indicators=lambda *args, **kwargs: pd.DataFrame(),
    )
    database_manager_module.data_repository = dummy_repo
    database_manager_module.db_manager = object()

    technical_module = types.ModuleType("data.processors.technical_indicators")
    technical_module.technical_indicators = types.SimpleNamespace(
        calculate_all_indicators=lambda *args, **kwargs: None
    )

    sys.modules["data"] = data_module
    sys.modules["data.processors"] = processors_module
    sys.modules["data.processors.technical_indicators"] = technical_module
    sys.modules["data.storage"] = storage_module
    sys.modules["data.storage.database_manager"] = database_manager_module

if "strategies.base.signal_generator" not in sys.modules:
    signal_generator_module = types.ModuleType("strategies.base.signal_generator")

    class _DummySignalGenerator:
        @staticmethod
        def crossover_signals(*args, **kwargs):  # pragma: no cover - testing stub
            return []

        @staticmethod
        def volume_confirmation(signals, *args, **kwargs):
            return signals

        @staticmethod
        def trend_filter(signals, *args, **kwargs):
            return signals

    class _DummySignalCondition:
        CROSSOVER = "crossover"
        CROSSUNDER = "crossunder"

    signal_generator_module.SignalGenerator = _DummySignalGenerator
    signal_generator_module.SignalCondition = _DummySignalCondition

    sys.modules["strategies.base.signal_generator"] = signal_generator_module

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation import BusinessStrategyEvaluator
from strategies import ChinaLimitUpConfig, ChinaLimitUpMomentumStrategy


def build_sample_dataset() -> pd.DataFrame:
    """Construct a miniature dataset emulating limit-up scenarios."""

    dates = pd.date_range("2024-01-08", periods=6, freq="B")

    def stock_frame(ts_code: str, base_price: float, pct_series: list[float], volumes: list[int], turnover: list[float]) -> pd.DataFrame:
        close_prices = [base_price]
        pre_close_list = [base_price * (1 - pct_series[0])]
        for pct in pct_series:
            next_price = close_prices[-1] * (1 + pct)
            close_prices.append(next_price)
            pre_close_list.append(close_prices[-2])
        close_prices = close_prices[1:]
        pre_close_list = pre_close_list[1:]

        data = pd.DataFrame(
            {
                "ts_code": ts_code,
                "trade_date": dates,
                "open": [pc * (1 + min(pct * 0.12, 0.04)) for pc, pct in zip(pre_close_list, pct_series)],
                "high": [pc * (1 + pct * 1.01) for pc, pct in zip(pre_close_list, pct_series)],
                "low": [pc * (1 + pct * 0.18) for pc, pct in zip(pre_close_list, pct_series)],
                "close": close_prices,
                "pre_close": pre_close_list,
                "vol": volumes,
                "turnover_rate": turnover,
            }
        )

        data["pct_chg"] = [pct * 100 for pct in pct_series]

        return data

    main_board = stock_frame(
        "600519.SH",
        base_price=1600,
        pct_series=[0.021, 0.035, 0.099, 0.101, 0.098, 0.099],
        volumes=[900000, 1100000, 1800000, 2400000, 3200000, 4200000],
        turnover=[0.9, 1.7, 7.5, 9.5, 11.0, 12.5],
    )

    growth_board = stock_frame(
        "300750.SZ",
        base_price=400,
        pct_series=[0.031, 0.042, 0.12, 0.185, 0.199, 0.198],
        volumes=[2000000, 2400000, 3600000, 5000000, 7200000, 9000000],
        turnover=[3.5, 5.3, 15.2, 18.8, 22.0, 23.5],
    )

    dataset = pd.concat([main_board, growth_board], ignore_index=True)
    return dataset


def main() -> None:
    dataset = build_sample_dataset()

    config = ChinaLimitUpConfig(
        name="China Limit Up Momentum",
        lookback_period=60,
        max_stocks=5,
        stop_loss_pct=0.08,
        take_profit_pct=0.16,
        max_position_size=0.2,
    )

    strategy = ChinaLimitUpMomentumStrategy(config)

    evaluator = BusinessStrategyEvaluator()
    report = evaluator.evaluate(strategy, dataset)

    print("Signals:")
    for signal in strategy.apply_risk_management(strategy.generate_signals(dataset)):
        print(signal)

    print("\nEvaluation report:")
    print(f"Score: {report['score']}")
    print(f"Passed: {report['passed']}")
    print("Breakdown:")
    for field_name, weight in evaluator.weights.items():
        component_score = getattr(report["breakdown"], field_name)
        print(f"  {field_name}: {component_score:.2f} / {weight}")

    if report["score"] < evaluator.min_passing_score:
        raise SystemExit("策略未达到商业级别门槛，需要继续优化。")


if __name__ == "__main__":
    main()
