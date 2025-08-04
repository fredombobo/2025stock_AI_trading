# 2025stock_AI_trading

A modular quantitative trading framework for the A-share market. The project aims to integrate data ingestion, factor engineering, strategy research and execution into a unified system.

## Local configuration
1. Copy `.env.example` to `.env` in the project root.
2. Edit `.env` and provide values for the required variables such as `TUSHARE_TOKEN`.
3. The `.env` file is listed in `.gitignore` and will not be committed.

## Documentation
- [Development Plan](docs/development_plan.md) – high-level roadmap of upcoming milestones.
- [Quant Trading System Overview](quant_trading_system.md) – detailed architecture notes.

## Example Strategy
A simple moving-average crossover strategy is provided under `strategies/momentum/ma_crossover.py` for reference.
