# DeepAlpha-V1: Machine Learning Quantitative Trading System (机器学习量化交易系统)

## Project Overview (项目概览)
DeepAlpha-V1 is an end-to-end automated Quantitative Trading (量化交易) system based on Machine Learning (机器学习). Designed specifically for the S&P 500 equity universe, it utilizes a gradient boosting framework (梯度提升框架) to extract non-linear Alpha (非线性超额收益) from market data. 

The system operates with a strict Out-of-Sample (样本外) validation process to prevent Data Leakage (数据泄露) and employs rigorous Risk Management (风险管理) through Probability Thresholding (概率阈值过滤).

## System Architecture (系统架构)
The workflow is completely decoupled into the following core modules:

* **Data Pipeline (数据流水线)** - `data_loader.py`
  Fetches End-of-Day (日终) price and volume data for S&P 500 constituents, handling data cleaning and formatting automatically.
* **Feature Engineering (特征工程)** - `factor_miner.py`
  Constructs advanced technical indicators, including Time-Series Momentum (时间序列动量), Volatility (波动率), and Cross-Sectional Ranking (横截面排序) factors.
* **Model Training (模型训练)** - `train_and_save.py`
  Trains a LightGBM classifier with Hyperparameter Tuning (超参数调优) and serializes (序列化) the optimal model for production.
* **Backtest Engine (回测引擎)** - `backtest_engine.py`
  A Vectorized Backtester (向量化回测框架) that simulates real-world trading, accounting for Transaction Costs (交易摩擦成本) and Overlapping Portfolios (重叠持仓).
* **Daily Inference (每日推理)** - `daily_predictor.py`
  Loads the persisted model to evaluate the latest Cross-Section (截面) data, outputting actionable trading signals.
* **Pipeline Orchestration (流水线调度)** - `run_pipeline.py`
  The master controller that executes the daily automated workflow sequentially with built-in Fault Tolerance (容错机制).

## Key Performance Indicators (核心绩效指标)
Based on strictly segregated Test Set (测试集) backtesting (2024-01 to Present):

* **Annualized Return (年化收益率)**: ~14.0%
* **Maximum Drawdown (最大回撤)**: < 4.0%
* **Sharpe Ratio (夏普比率)**: 1.67
* **Optimal Probability Threshold (最优置信度阈值)**: 0.65

## Getting Started (快速开始)

### Prerequisites (环境要求)
* Python 3.12+
* Required packages: `pandas`, `numpy`, `lightgbm`, `scikit-learn`, `matplotlib`, `joblib`, `lxml`

### Daily Operation (日常运行)
To execute the fully automated daily trading pipeline after market close:
```bash
python run_pipeline.py