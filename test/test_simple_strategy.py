import pandas as pd
from core_strategy import SimpleMovingAverageStrategy, SignalType


def test_simple_ma_strategy_generates_buy_signal():
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    prices = [10] * 20 + [20] * 20
    df = pd.DataFrame({
        "ts_code": ["TEST"] * 40,
        "trade_date": dates,
        "close_price": prices,
    })

    strategy = SimpleMovingAverageStrategy(short_window=5, long_window=20)
    signals = strategy.generate_signals(df)

    assert any(sig.signal_type == SignalType.BUY for sig in signals)
