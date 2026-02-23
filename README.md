# DeepAlpha-V1: Machine Learning Quantitative Trading System

## Project Overview
DeepAlpha-V1 is an end-to-end automated Quantitative Trading system based on Machine Learning. Designed specifically for the S&P 500 equity universe, it utilizes a gradient boosting framework to extract non-linear Alpha from market data.

The system operates with a strict Out-of-Sample validation process to prevent Data Leakage and employs rigorous Risk Management through Probability Thresholding.

## System Architecture
The workflow is completely decoupled into the following core modules:

* **Data Pipeline** - `data_loader.py`
  Fetches End-of-Day price and volume data for S&P 500 constituents, handling data cleaning and formatting automatically.
* **Feature Engineering** - `factor_miner.py`
  Constructs advanced technical indicators, including Time-Series Momentum, Volatility, and Cross-Sectional Ranking factors.
* **Model Training** - `train_and_save.py`
  Trains a LightGBM classifier with Hyperparameter Tuning and serializes the optimal model for production.
* **Backtest Engine** - `backtest_engine.py`
  A Vectorized Backtester that simulates real-world trading, accounting for Transaction Costs and Overlapping Portfolios.
* **Daily Inference** - `daily_predictor.py`
  Loads the persisted model to evaluate the latest Cross-Section data, outputting actionable trading signals.
* **Pipeline Orchestration** - `run_pipeline.py`
  The master controller that executes the daily automated workflow sequentially with built-in Fault Tolerance.

## Key Performance Indicators
Based on strictly segregated Test Set backtesting (2024-01 to Present):

* **Annualized Return**: ~14.0%
* **Maximum Drawdown**: < 4.0%
* **Sharpe Ratio**: 1.67
* **Optimal Probability Threshold**: 0.65

## Getting Started

### Prerequisites
* Python 3.12+
* Required packages: `pandas`, `numpy`, `lightgbm`, `scikit-learn`, `matplotlib`, `joblib`, `lxml`

### Daily Operation
To execute the fully automated daily trading pipeline after market close:
```bash
python run_pipeline.py


