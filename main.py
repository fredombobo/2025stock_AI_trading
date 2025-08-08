import pandas as pd
from core_strategy import SimpleMovingAverageStrategy


def generate_sample_data() -> pd.DataFrame:
    """Generate sample price data with a clear MA crossover."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    prices = [100] * 30 + [110] * 30
    df = pd.DataFrame({
        "ts_code": ["TEST"] * 60,
        "trade_date": dates,
        "close_price": prices,
    })
    return df


def main():
    data = generate_sample_data()
    strategy = SimpleMovingAverageStrategy(short_window=5, long_window=20)
    signals = strategy.generate_signals(data)
    for signal in signals:
        print(signal.to_dict())


if __name__ == "__main__":
    main()
