import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 配置
# ==========================================
INPUT_FILE = 'data/sp500_data.parquet'
OUTPUT_FILE = 'data/sp500_factors.parquet'

# ==========================================
# 2. 时间序列特征 (按单只股票计算)
# ==========================================
def compute_time_series_factors(df):
    # A. 基础动量与均线
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['Bias_50'] = (df['close'] / df['EMA_50']) - 1

    # B. RSI (14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['RSI_14'] = df['RSI_14'].fillna(100)

    # C. MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # D. 波动率 (NATR)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
    df['NATR_14'] = df['ATR_14'] / df['close']

    # ----------------------------------------------------
    # [新增] E. 中长周期动量因子 (Momentum Returns)
    # ----------------------------------------------------
    df['Ret_5D'] = df['close'].pct_change(5)
    df['Ret_20D'] = df['close'].pct_change(20)
    df['Ret_60D'] = df['close'].pct_change(60)

    # ----------------------------------------------------
    # [新增] F. 价格极值锚点 (Price Anchors)
    # ----------------------------------------------------
    # 距离过去250天(一年)最高点和最低点的百分比
    rolling_high_250 = df['high'].rolling(window=250, min_periods=1).max()
    rolling_low_250 = df['low'].rolling(window=250, min_periods=1).min()
    df['Dist_High_250'] = (df['close'] / rolling_high_250) - 1
    df['Dist_Low_250'] = (df['close'] / rolling_low_250) - 1

    # 生成预测标签 (未来5天收益率)
    df['Target_5D'] = df['close'].pct_change(5).shift(-5)
    df['Target_Dir'] = (df['Target_5D'] > 0).astype(int)

    return df

# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"[Error] 输入文件不存在: {INPUT_FILE}")
        exit()

    print("[1/3] 读取 Parquet 原始数据...")
    df = pd.read_parquet(INPUT_FILE)
    df_reset = df.reset_index()
    
    print("[2/3] 计算时间序列特征 (纵向计算)...")
    # 按 ticker 分组计算单只股票的独立特征
    factors_df = df_reset.groupby('ticker', group_keys=False).apply(compute_time_series_factors)
    
    print("[3/3] 计算横截面排名特征 (横向计算)...")
    # 按 date 分组，计算每天同一时刻所有股票的相对排名 (百分位 0.0 ~ 1.0)
    # pct=True 表示转换为百分位排名，值越大表示在当天全市场中排名越靠前
    rank_cols = ['RSI_14', 'Ret_20D', 'Ret_60D', 'Dist_High_250']
    for col in rank_cols:
        rank_col_name = f'Rank_{col}'
        factors_df[rank_col_name] = factors_df.groupby('date')[col].rank(pct=True)

    # 清洗空值行（均线计算、长周期收益率以及标签平移产生的空值）
    original_len = len(factors_df)
    factors_df = factors_df.dropna()
    print(f"[Clean] 剔除空值行: {original_len} -> {len(factors_df)}")
    
    # 恢复多重索引
    if 'date' in factors_df.columns and 'ticker' in factors_df.columns:
        factors_df = factors_df.set_index(['date', 'ticker']).sort_index()
        
    # 保存覆盖原文件
    factors_df.to_parquet(OUTPUT_FILE)
    
    print(f"\n[Success] 因子挖掘 V2 完成！已保存至: {OUTPUT_FILE}")
    print(f"[Info] 当前特征矩阵总列数: {factors_df.shape[1]}")
    
    # 打印验证横截面排名
    print("\n[Preview] 新增横截面排名因子示例 (最后 5 行):")
    preview_cols = ['close', 'Ret_20D', 'Rank_Ret_20D', 'Rank_Dist_High_250']
    print(factors_df[preview_cols].tail())
