
---

# DeepAlpha-V1
**一个基于机器学习的端到端 S&P 500 量化交易系统**
**An End-to-End Machine Learning Quantitative Trading System for S&P 500**

---

## 项目简介 / Project Overview

DeepAlpha-V1 是一个**实用、模块化、可自动运行**的量化交易测试框架。
系统专注于 S&P 500 成分股，使用 LightGBM 梯度提升树提取非线性 Alpha 信号，严格防止数据泄漏，并通过**概率阈值风控**实现高确定性交易。

核心理念：**只在模型非常有把握（概率 ≥ 0.65）时才出手，否则坚决空仓**，追求**高夏普 + 低回撤**。

**English**:
DeepAlpha-V1 is a practical, modular, and fully automated quantitative trading framework for S&P 500 stocks. It uses LightGBM to extract non-linear alpha signals with strict out-of-sample validation and probability-based risk control.

**Philosophy**:
Trade only with high confidence (prob ≥ 0.65), otherwise stay in cash — aiming for high Sharpe and low drawdown.

---

## 主要特性 / Key Features

* ✅ 完整端到端流水线（数据 → 因子 → 模型 → 预测 → 回测）
* ✅ 严格 Out-of-Sample 测试（2024-01 之后作为测试集）
* ✅ 时间序列 + 横截面排名因子（相对强弱）
* ✅ **LightGBM 二分类模型（预测未来 1 日涨跌）**
* ✅ **强制时序排序，彻底防止数据穿越 (Data Leakage)**
* ✅ 概率阈值严格风控（默认 0.65），最多持仓 5 只，等权重
* ✅ **真实每日调仓回测，精准双边换手率与摩擦成本计算**
* ✅ 考虑交易成本（0.2% 单边）和持仓重叠
* ✅ 每日一键自动化运行（`run_pipeline.py`）
* ✅ 支持 SHAP 解释（全局特征重要性 + 单股票局部贡献）
* ✅ 向量化回测引擎 + 多阈值对比图表

* ✅ Full end-to-end pipeline (data → factors → model → prediction → backtest)
* ✅ Strict out-of-sample testing (data after 2024-01 used as test set)
* ✅ Time-series + cross-sectional ranking factors (relative strength)
* ✅ **LightGBM binary classification model (predicting 1-day future returns)**
* ✅ **Enforced time-series sorting to eliminate data leakage**
* ✅ Probability-threshold-based risk control (default 0.65), max 5 positions, equal-weighted
* ✅ **True daily rebalancing with precise two-way turnover and friction cost calculation**
* ✅ Transaction costs considered (0.2% one-way) and overlapping positions handled
* ✅ One-click daily automation (run_pipeline.py)
* ✅ SHAP interpretability support (global feature importance + per-stock local contributions)
* ✅ Vectorized backtesting engine with multi-threshold comparison plots

---

## 回测表现（2024-01 ~ 2026-04） / Backtest Performance

使用严格时间分割的 Out-of-Sample 回测结果（**真实每日调仓 + 精准双边换手成本**）：

| 概率阈值 Threshold | 交易次数 Trades | 总收益率 Total Return | 年化换手率 Ann. Turnover | 交易损耗 Trans. Cost | 最大回撤 Max DD | 夏普比率 Sharpe |
| -------------- | ----------- | ----------------- | ------------------- | ----------------- | ------------- | ----------- |
| 0.55           | 2825        | 90.52%            | 214.9x              | 203.0%            | -28.51%       | 2.52        |
| 0.60           | 1672        | 81.72%            | 126.8x              | 119.8%            | -27.48%       | 1.88        |
| **0.65（推荐）**   | **467**     | **128.03%**       | **35.4x**           | **33.4%**         | **-14.06%**   | **2.49**    |
| 0.70           | 210         | 25.00%            | 15.9x               | 15.0%             | -9.48%        | 1.33        |
| 0.75           | 78          | 23.57%            | 5.9x                | 5.6%              | -1.15%        | 1.59        |

**关键改进 / Key Improvements**：
* **预测周期**：从 5 日改为 1 日，支持真实每日调仓
* **防穿越**：强制时序排序，消除 groupby 乱序风险
* **精准成本**：基于实际持仓变化的双边换手率计算，非粗略估算

**结论 / Conclusion**：
**0.65** 是收益与风险的最佳平衡点，在扣除高额交易成本后仍显著优于标普 500 基准。
高阈值 (0.75) 策略虽然收益较低，但换手率仅 5.9 倍，交易成本仅 5.6%，适合大资金实盘。

---

## 项目结构 / Project Structure

```bash
DeepAlpha-V1/
├── data/                      # 数据和因子文件 / Data & factors
├── models/                    # 保存的 LightGBM 模型 / Saved models
├── backtest_thresholds.png    # 回测资金曲线图 / Equity curve
├── data_loader.py             # 下载 S&P500 数据
├── factor_miner.py            # 因子工程
├── train_and_save.py          # 训练并保存模型
├── daily_predictor.py         # 每日预测 + SHAP 解释
├── backtest_engine.py         # 回测引擎
├── run_pipeline.py            # 一键自动流水线
└── README.md
```

---

## 快速开始 / Quick Start

### 1. 环境准备 / Environment Setup

```bash
pip install pandas numpy lightgbm scikit-learn matplotlib joblib lxml yfinance requests shap
```

---

### 2. 首次运行（推荐顺序） / First Run (Recommended Order)

```bash
python data_loader.py       # 下载最新 S&P500 数据
python factor_miner.py      # 生成因子
python train_and_save.py    # 训练模型（使用全量数据）
python daily_predictor.py   # 查看交易信号（含 SHAP 解释）
```

**一键自动运行（推荐日常使用） / Daily One-Click Run:**

```bash
python run_pipeline.py
```

---

### 3. 运行回测 / Run Backtest

```bash
python backtest_engine.py
```

---

## 日常使用建议 / Daily Usage Tips

* 每个交易日美股收盘后运行 `run_pipeline.py` 或 `daily_predictor.py`
* 当输出“无高确定性交易机会”时 → **严格保持空仓**
* 建议每月重新运行 `train_and_save.py`，适应市场变化
* 可通过修改 `daily_predictor.py` 中的 `OPTIMAL_THRESHOLD` 调整策略激进程度（建议保持 0.65）
* **实盘注意**：高阈值策略 (≥0.75) 换手率低，更适合大资金；低阈值策略 (<0.60) 换手率高，需警惕交易成本侵蚀利润


* Run run_pipeline.py or daily_predictor.py after each U.S. market close
* When the output indicates “no high-confidence trading opportunities” → strictly stay in cash
* **Live Trading Note**: High-threshold strategies (≥0.75) have low turnover, suitable for large capital; low-threshold strategies (<0.60) have high turnover, beware of transaction cost erosion

* Retrain the model monthly via train_and_save.py to adapt to evolving market regimes
* Adjust strategy aggressiveness by modifying OPTIMAL_THRESHOLD in daily_predictor.py (recommended: keep at 0.65)


---

## 注意事项 / Disclaimer

* 本项目仅供学习和量化研究使用，不构成任何投资建议。
* 历史回测表现不代表未来收益。
* 实盘交易需考虑滑点、流动性、税费等额外成本。
* 市场有风险，投资需谨慎。

* This project is for educational and quantitative research purposes only → not investment advice
* Historical backtest performance → does not guarantee future returns
* Live trading must consider → slippage, liquidity, transaction costs, taxes
* Markets involve risk → invest with caution