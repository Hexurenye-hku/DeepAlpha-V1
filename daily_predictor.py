import pandas as pd
import joblib
import os

INPUT_FILE = 'data/sp500_factors.parquet'
MODEL_PATH = 'models/optimal_lgbm.pkl'

# 锁定最优阈值
OPTIMAL_THRESHOLD = 0.65
MAX_POSITIONS = 5

def predict_tomorrow():
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] 找不到模型文件 {MODEL_PATH}，请先运行 train_and_save.py")
        return

    print("[1/3] 加载最新因子数据与底层模型...")
    df = pd.read_parquet(INPUT_FILE)
    df = df.reset_index()
    
    model = joblib.load(MODEL_PATH)

    # 提取时间轴上最新的一天作为截面数据
    latest_date = df['date'].max()
    if pd.isna(latest_date):
        print("[Error] 日期列为空，请检查数据。")
        return
        
    # 格式化日期显示，兼容字符串或 datetime 对象
    date_str = latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date)
    print(f"当前截面数据日期: {date_str}")

    latest_df = df[df['date'] == latest_date].copy()

    exclude_cols = ['Target_5D', 'Target_Dir', 'close', 'high', 'low', 'open', 'volume', 'date', 'ticker']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X_latest = latest_df[feature_cols]

    print("[2/3] 执行模型推理计算...")
    latest_df['Prob_Up'] = model.predict_proba(X_latest)[:, 1]

    print(f"[3/3] 应用严格风控阈值 (> {OPTIMAL_THRESHOLD}) 进行筛选...")
    signals = latest_df[latest_df['Prob_Up'] > OPTIMAL_THRESHOLD]

    print("\n=======================================================")
    print("               DeepAlpha-V1 明日实盘交易清单               ")
    print("=======================================================")
    
    if len(signals) > 0:
        top_picks = signals.nlargest(MAX_POSITIONS, 'Prob_Up')
        
        # 格式化输出表格
        output = top_picks[['ticker', 'Prob_Up', 'close']].copy()
        output['Prob_Up'] = output['Prob_Up'].apply(lambda x: f"{x:.2%}")
        output['close'] = output['close'].apply(lambda x: f"${x:.2f}")
        output.columns = ['股票代码 (Ticker)', '模型确信度 (Probability)', '截面收盘价 (Close)']
        
        print(output.to_string(index=False))
        print("-------------------------------------------------------")
        print("操作建议: 在次日开盘时对上述标的进行等权重买入，并设定 5 日后平仓。")
    else:
        print(f"模型判定当前截面无高确定性交易机会 (全部标的概率均低于 {OPTIMAL_THRESHOLD})。")
        print("操作建议: 保持空仓或仅持有现有底仓，严格规避市场噪音。")
        
    print("=======================================================")

if __name__ == "__main__":
    predict_tomorrow()