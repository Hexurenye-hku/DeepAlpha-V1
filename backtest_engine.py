import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import os

INPUT_FILE = 'data/sp500_factors.parquet'
OUTPUT_IMAGE = 'backtest_thresholds.png'

def run_grid_search():
    print("[1/4] 读取数据与特征工程...")
    df = pd.read_parquet(INPUT_FILE)
    
    # === 任务 1: 特征工程提纯 (Feature Purification) ===
    # 严格白名单/黑名单机制，剥离市场 Beta，仅保留选股 Alpha
    # Strict Whitelist/Blacklist mechanism to strip Market Beta and keep only Stock Selection Alpha
    
    # 黑名单：剔除绝对价格、绝对均线、绝对收益率、原始量价
    # Blacklist: Exclude absolute prices, moving averages, returns, and raw OHLCV
    blacklist = [
        'Target_5D', 'Target_Dir', 'Target_1D', 
        'close', 'high', 'low', 'open', 'volume',
        'EMA_10', 'EMA_50', 
        'Ret_5D', 'Ret_20D', 'Ret_60D',
        'Dist_High_250', 'Dist_Low_250'
    ]
    
    # 白名单筛选逻辑：
    # 1. 以 Rank_ 开头的横截面排名特征 (Cross-sectional Rank Features)
    # 2. 无量纲的相对指标 (Dimensionless Relative Indicators)
    whitelist_prefix = ['Rank_']
    whitelist_relative = ['Bias_50', 'NATR_14', 'MACD_hist', 'RSI_14']
    
    feature_cols = []
    for col in df.columns:
        if col in blacklist:
            continue
        # 检查是否在白名单前缀或具体相对指标中
        is_rank_feature = any(col.startswith(prefix) for prefix in whitelist_prefix)
        is_relative_feature = col in whitelist_relative
        
        if is_rank_feature or is_relative_feature:
            feature_cols.append(col)
    
    # 打印最终入选特征以供核对
    # Print final selected features for verification
    print(f"   [Feature Purification] 最终入选 {len(feature_cols)} 个特征:")
    print(f"   {feature_cols}")
    
    exclude_cols = ['Target_5D', 'Target_Dir', 'Target_1D', 'close', 'high', 'low', 'open', 'volume']

    print("[2/4] 训练 LightGBM 模型 (多核并行)...")
    df = df.reset_index()
    train_df = df[df['date'] < '2024-01-01']
    test_df = df[df['date'] >= '2024-01-01'].copy()

    X_train = train_df[feature_cols]
    y_train = train_df['Target_Dir']  # 仍使用 5 日方向作为训练标签
    X_test = test_df[feature_cols]

    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("[3/4] 生成预测信号...")
    test_df['Prob_Up'] = model.predict_proba(X_test)[:, 1]

    print("[4/4] 执行阈值网格搜索 (Grid Search) - 每日等权调仓模式...")
    
    # 设定需要遍历的概率阈值列表
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
    MAX_POSITIONS = 5
    COST_RATE = 0.002  # 单边成本 0.2%
    BUFFER = 0.05      # 持仓缓冲带 5% (Hysteresis Buffer / Position Stickiness)
    
    results = []
    
    plt.style.use('default')
    plt.figure(figsize=(12, 6))
    
    # === 计算大盘基准线 (使用 1 日收益) ===
    # Market benchmark using true daily returns
    daily_bench = []
    for date, group in test_df.groupby('date'):
        daily_bench.append({'date': date, 'bench_ret': group['Target_1D'].mean()})
    bench_df = pd.DataFrame(daily_bench).set_index('date').sort_index()
    bench_df['cumulative_ret'] = (1 + bench_df['bench_ret']).cumprod()
    plt.plot(bench_df.index, bench_df['cumulative_ret'], label='Market Benchmark', color='gray', linestyle='--')
    
    # 遍历不同的确信度阈值
    for threshold in thresholds:
        daily_records = []
        trade_count = 0
        
        # 换手率计算相关变量
        prev_positions = set()  # T-1 日的持仓股票代码集合
        total_turnover = 0.0    # 累计双边换手次数 (cumulative two-way turnover)
        
        for date, group in test_df.groupby('date'):
            # === 任务 2: 引入持仓缓冲带机制 (Hysteresis Buffer / Position Stickiness) ===
            # 防止股票在阈值边缘微小波动时频繁换手
            # Prevent frequent turnover when stocks fluctuate marginally around the threshold
            
            # 1. 新买入条件：Prob_Up > threshold
            # New Entry: Must satisfy Prob_Up > threshold
            new_entry_signals = group[group['Prob_Up'] > threshold]
            
            # 2. 继续持有条件：已在持仓中且 Prob_Up > threshold - BUFFER
            # Retention: Already in portfolio AND Prob_Up > threshold - BUFFER
            retention_signals = group[
                (group['ticker'].isin(prev_positions)) & 
                (group['Prob_Up'] > (threshold - BUFFER))
            ]
            
            # 3. 合并新老股票，按 Prob_Up 降序排列，截取前 MAX_POSITIONS 只
            # Merge new and retained stocks, sort by Prob_Up descending, take top MAX_POSITIONS
            if len(new_entry_signals) > 0 or len(retention_signals) > 0:
                # 合并并去重 (merge and deduplicate)
                combined_signals = pd.concat([new_entry_signals, retention_signals]).drop_duplicates(subset='ticker')
                
                # 按概率降序排序 (sort by probability descending)
                top_picks = combined_signals.nlargest(MAX_POSITIONS, 'Prob_Up')
                
                # 获取最终持仓代码集合 (get final position set)
                current_positions = set(top_picks['ticker'].tolist())
                
                # === 每日真实收益计算 (True Daily Return) ===
                # 计算这 N 只股票当天的真实 1 日平均收益
                # Calculate the true 1-day average return of the selected basket
                daily_gross_return = top_picks['Target_1D'].mean()
                
                # === 精准摩擦成本扣除 (Precise Friction Deduction) ===
                # 计算当日实际换手比例 (Turnover Ratio)
                # Formula: Turnover_t = |Position_t - Position_{t-1}| / 2
                # 对称差集元素个数 / (2 * 最大持仓数) = 归一化换手比例
                positions_changed = len(current_positions.symmetric_difference(prev_positions))
                turnover_ratio = positions_changed / (2.0 * MAX_POSITIONS)
                
                # 当日净收益 = 毛收益 - (换手比例 * 2 * 单边成本)
                # Net Return = Gross Return - (Turnover Ratio * 2 * Single-side Cost)
                # 注意：换手部分必须扣除双边成本 (Round-trip Cost)
                daily_net_return = daily_gross_return - (turnover_ratio * 2 * COST_RATE)
                
                # 更新持仓记录
                prev_positions = current_positions
                
                daily_records.append({'date': date, 'strategy_ret': daily_net_return})
                trade_count += len(top_picks)
            else:
                # 无信号时，持仓清空，计算清仓换手成本
                # When no signal, clear all positions and calculate liquidation turnover
                positions_changed = len(prev_positions)
                turnover_ratio = positions_changed / (2.0 * MAX_POSITIONS)
                liquidation_cost = turnover_ratio * 2 * COST_RATE
                
                total_turnover += turnover_ratio
                prev_positions = set()
                
                # 清仓日收益为负的摩擦成本
                daily_records.append({'date': date, 'strategy_ret': -liquidation_cost})

        bt_df = pd.DataFrame(daily_records).set_index('date').sort_index()
        bt_df['cumulative_ret'] = (1 + bt_df['strategy_ret']).cumprod()
        
        # 回撤计算
        bt_df['high_water_mark'] = bt_df['cumulative_ret'].cummax()
        bt_df['drawdown'] = (bt_df['cumulative_ret'] / bt_df['high_water_mark']) - 1
        max_drawdown = bt_df['drawdown'].min()
        
        # 收益与夏普比率计算
        total_return = bt_df['cumulative_ret'].iloc[-1] - 1
        annual_return = total_return * (252 / len(bt_df))
        annual_volatility = bt_df['strategy_ret'].std() * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.03) / annual_volatility if annual_volatility != 0 else 0
        
        # === 换手率与交易损耗计算 (Turnover & Transaction Cost Calculation) ===
        # 年化换手率 = 总换手比例 * (252 / 交易日数)
        # Annualized Turnover = Total Turnover Ratio * (252 / Trading Days)
        trading_days = len(bt_df)
        annual_turnover = total_turnover * (252 / trading_days)
        
        # 总交易损耗 = 总换手比例 * 2 * 单边成本
        # Total Transaction Cost = Total Turnover Ratio * 2 * Single-side Cost
        total_transaction_cost = total_turnover * 2 * COST_RATE
        
        # 记录该阈值下的表现
        results.append({
            'Threshold': threshold,
            'Trades': trade_count,
            'Total Return': f"{total_return:.2%}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Sharpe': round(sharpe_ratio, 2),
            'Ann. Turnover': f"{annual_turnover:.1f}x",
            'Trans. Cost': f"{total_transaction_cost:.2%}"
        })
        
        # 绘制资金曲线
        plt.plot(bt_df.index, bt_df['cumulative_ret'], label=f'Threshold: {threshold}')
    
    print("\n=======================================================")
    print("                 阈值参数网格搜索报告                  ")
    print("=======================================================")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("=======================================================")
    
    plt.title('Equity Curves by Probability Threshold', fontsize=14, fontweight='bold')
    plt.ylabel('Cumulative Net Value')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"\n操作完成。对比图表已保存至: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"[Error] 找不到输入文件: {INPUT_FILE}")
    else:
        run_grid_search()