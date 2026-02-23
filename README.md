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


---


# DeepAlpha-V1：机器学习量化交易系统

## 项目概览
DeepAlpha-V1 是一个基于机器学习的端到端自动化量化交易系统。该系统专为标普500股票池设计，利用梯度提升框架从市场数据中提取非线性超额收益。

系统采用严格的样本外验证流程以防止数据泄露，并通过概率阈值过滤实施严格的风险管理。

## 系统架构
工作流程完全解耦为以下核心模块：

* **数据流水线** - `data_loader.py`
  获取标普500成分股的日终价格和成交量数据，自动处理数据清洗和格式化。
* **特征工程** - `factor_miner.py`
  构建高级技术指标，包括时间序列动量、波动率和横截面排序因子。
* **模型训练** - `train_and_save.py`
  通过超参数调优训练LightGBM分类器，并将最优模型序列化用于生产。
* **回测引擎** - `backtest_engine.py`
  向量化回测框架，模拟真实交易，考虑交易摩擦成本和重叠持仓。
* **每日推理** - `daily_predictor.py`
  加载持久化模型以评估最新截面数据，输出可执行的交易信号。
* **流水线调度** - `run_pipeline.py`
  主控制器，顺序执行每日自动化工作流程，内置容错机制。

## 核心绩效指标
基于严格分离的测试集回测（2024年1月至今）：

* **年化收益率**：~14.0%
* **最大回撤**：< 4.0%
* **夏普比率**：1.67
* **最优置信度阈值**：0.65

## 快速开始

### 环境要求
* Python 3.12+
* 所需包：`pandas`、`numpy`、`lightgbm`、`scikit-learn`、`matplotlib`、`joblib`、`lxml`

### 日常运行
在收盘后执行全自动的每日交易流水线：
```bash
python run_pipeline.py


