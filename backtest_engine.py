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
    
    exclude_cols = ['Target_5D', 'Target_Dir', 'close', 'high', 'low', 'open', 'volume']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print("[2/4] 训练 LightGBM 模型 (多核并行)...")
    df = df.reset_index()
    train_df = df[df['date'] < '2024-01-01']
    test_df = df[df['date'] >= '2024-01-01'].copy()

    X_train = train_df[feature_cols]
    y_train = train_df['Target_Dir']
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

    print("[4/4] 执行阈值网格搜索 (Grid Search)...")
    
    # 设定需要遍历的概率阈值列表
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
    MAX_POSITIONS = 5
    COST_RATE = 0.002
    HOLDING_DAYS = 5
    
    results = []
    
    plt.style.use('default')
    plt.figure(figsize=(12, 6))
    
    # 计算大盘基准线
    daily_bench = []
    for date, group in test_df.groupby('date'):
        daily_bench.append({'date': date, 'bench_ret': group['Target_5D'].mean() / HOLDING_DAYS})
    bench_df = pd.DataFrame(daily_bench).set_index('date').sort_index()
    bench_df['cumulative_ret'] = (1 + bench_df['bench_ret']).cumprod()
    plt.plot(bench_df.index, bench_df['cumulative_ret'], label='Market Benchmark', color='gray', linestyle='--')
    
    # 遍历不同的确信度阈值
    for threshold in thresholds:
        daily_records = []
        trade_count = 0
        
        for date, group in test_df.groupby('date'):
            signals = group[group['Prob_Up'] > threshold]
            
            if len(signals) > 0:
                top_picks = signals.nlargest(MAX_POSITIONS, 'Prob_Up')
                basket_return = top_picks['Target_5D'].mean() - COST_RATE
                daily_contribution = basket_return / HOLDING_DAYS
                daily_records.append({'date': date, 'strategy_ret': daily_contribution})
                trade_count += len(top_picks)
            else:
                daily_records.append({'date': date, 'strategy_ret': 0.0})

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
        
        # 记录该阈值下的表现
        results.append({
            'Threshold': threshold,
            'Trades': trade_count,
            'Total Return': f"{total_return:.2%}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Sharpe': round(sharpe_ratio, 2)
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