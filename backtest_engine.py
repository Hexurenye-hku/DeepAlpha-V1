import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import os

INPUT_FILE = 'data/sp500_factors.parquet'
OUTPUT_IMAGE = 'backtest_result.png'

def run_backtest():
    print("[1/5] 读取数据与特征工程...")
    df = pd.read_parquet(INPUT_FILE)
    
    exclude_cols = ['Target_5D', 'Target_Dir', 'close', 'high', 'low', 'open', 'volume']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print("[2/5] 训练 LightGBM 模型 (多核并行)...")
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

    print("[3/5] 生成预测信号...")
    test_df['Prob_Up'] = model.predict_proba(X_test)[:, 1]

    print("[4/5] 执行向量化回测 (计算策略与基准)...")
    
    PROB_THRESHOLD = 0.55
    MAX_POSITIONS = 5
    COST_RATE = 0.002
    HOLDING_DAYS = 5
    
    daily_records = []
    
    for date, group in test_df.groupby('date'):
        # 计算当天的基准收益 (假设等权买入测试集中所有的股票)
        bench_ret = (group['Target_5D'].mean()) / HOLDING_DAYS
        
        signals = group[group['Prob_Up'] > PROB_THRESHOLD]
        
        if len(signals) > 0:
            top_picks = signals.nlargest(MAX_POSITIONS, 'Prob_Up')
            basket_return = top_picks['Target_5D'].mean() - COST_RATE
            daily_contribution = basket_return / HOLDING_DAYS
            daily_records.append({
                'date': date, 
                'strategy_ret': daily_contribution,
                'bench_ret': bench_ret
            })
        else:
            daily_records.append({
                'date': date, 
                'strategy_ret': 0.0,
                'bench_ret': bench_ret
            })

    # 汇总计算
    bt_df = pd.DataFrame(daily_records).set_index('date').sort_index()
    bt_df['cumulative_ret'] = (1 + bt_df['strategy_ret']).cumprod()
    bt_df['bench_cumulative_ret'] = (1 + bt_df['bench_ret']).cumprod()
    
    # 回撤计算
    bt_df['high_water_mark'] = bt_df['cumulative_ret'].cummax()
    bt_df['drawdown'] = (bt_df['cumulative_ret'] / bt_df['high_water_mark']) - 1
    
    print("[5/5] 正在绘制并保存资金曲线图...")
    
    # 设置图表样式
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # 绘制净值曲线
    ax1.plot(bt_df.index, bt_df['cumulative_ret'], label='DeepAlpha-V1 Strategy', color='firebrick', linewidth=2)
    ax1.plot(bt_df.index, bt_df['bench_cumulative_ret'], label='Market Equal-Weight Benchmark', color='gray', linestyle='--')
    ax1.set_title('DeepAlpha-V1 Backtest Performance (2024 - Present)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Net Value')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 绘制回撤图
    ax2.fill_between(bt_df.index, bt_df['drawdown'], 0, color='firebrick', alpha=0.3)
    ax2.plot(bt_df.index, bt_df['drawdown'], color='firebrick', linewidth=1)
    ax2.set_ylabel('Drawdown')
    ax2.set_xlabel('Date')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"\n操作完成。图表已保存至项目根目录: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"[Error] 找不到输入文件: {INPUT_FILE}")
    else:
        run_backtest()