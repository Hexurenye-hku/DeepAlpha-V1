import pandas as pd
import joblib
import os
import shap
import numpy as np

INPUT_FILE = 'data/sp500_factors.parquet'
MODEL_PATH = 'models/optimal_lgbm.pkl'

OPTIMAL_THRESHOLD = 0.65
MAX_POSITIONS = 5

def predict_tomorrow():
    print("[1/3] 加载最新因子数据与底层模型...")
    df = pd.read_parquet(INPUT_FILE).reset_index()
    
    model = joblib.load(MODEL_PATH)

    print("   → 正在初始化 SHAP 解释器...")
    explainer = shap.TreeExplainer(model)

    latest_date = df['date'].max()
    date_str = latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date)
    print(f"当前截面数据日期: {date_str}")

    latest_df = df[df['date'] == latest_date].copy().reset_index(drop=True)

    exclude_cols = ['Target_5D', 'Target_Dir', 'close', 'high', 'low', 'open', 'volume', 'date', 'ticker']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X_latest = latest_df[feature_cols].values

    print("[2/3] 执行模型推理 + SHAP 计算...")
    latest_df['Prob_Up'] = model.predict_proba(X_latest)[:, 1]

    # SHAP 值处理（兼容不同版本）
    shap_values = explainer.shap_values(X_latest)
    if isinstance(shap_values, list):
        shap_values_up = shap_values[1]      # 二分类取正类（上涨）
    else:
        shap_values_up = shap_values         # 某些版本直接返回正类

    # 全局重要性（平均 |SHAP|）
    global_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(shap_values_up).mean(axis=0)
    }).sort_values('importance', ascending=False)

    print("\n【全局特征重要性 - Top 10】")
    print(global_importance.head(10).to_string(index=False))

    print(f"\n[3/3] 应用风控阈值 (> {OPTIMAL_THRESHOLD}) ...")

    signals = latest_df[latest_df['Prob_Up'] > OPTIMAL_THRESHOLD]

    print("\n" + "="*60)
    print("               DeepAlpha-V1 明日实盘交易清单")
    print("="*60)

    if len(signals) > 0:
        top_picks = signals.nlargest(MAX_POSITIONS, 'Prob_Up')
        # 输出表格...
        output = top_picks[['ticker', 'Prob_Up', 'close']].copy()
        output['Prob_Up'] = output['Prob_Up'].map("{:.2%}".format)
        output['close'] = output['close'].map("${:.2f}".format)
        output.columns = ['Ticker', '确信度', '收盘价']
        print(output.to_string(index=False))
        print("\n操作建议: 次日开盘等权重买入上述标的，持仓5天。")

        print("\n【局部解释 - 入选股票 Top 3 正向贡献特征】")
        for i, row in top_picks.iterrows():
            ticker = row['ticker']
            prob = row['Prob_Up']
            shap_row = shap_values_up[i]
            contrib = pd.DataFrame({'feature': feature_cols, 'shap': shap_row})
            contrib = contrib.sort_values('shap', ascending=False)
            print(f"\n▶ {ticker} (确信度 {prob:.2%})")
            print(contrib.head(3).to_string(index=False))

    else:
        print(f"模型判定当前截面无高确定性机会（全部概率 < {OPTIMAL_THRESHOLD}）。")
        print("操作建议: 保持空仓，严格规避噪音。")

        # 显示 Top 5 供观察（修复重复问题）
        print("\n【参考：当前 Top 5 最高概率股票 + 局部解释】")
        top_candidates = latest_df.nlargest(5, 'Prob_Up').reset_index(drop=True)
        for i in range(len(top_candidates)):
            row = top_candidates.iloc[i]
            ticker = row['ticker']
            prob = row['Prob_Up']
            shap_row = shap_values_up[top_candidates.index[i]]   # 使用 iloc 后的新索引
            contrib = pd.DataFrame({'feature': feature_cols, 'shap': shap_row})
            contrib = contrib.sort_values('shap', ascending=False)
            
            print(f"\n▶ {ticker} (确信度 {prob:.2%})")
            print(contrib.head(3).to_string(index=False))

    print("="*60)

if __name__ == "__main__":
    predict_tomorrow()