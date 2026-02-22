import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 配置
# ==========================================
INPUT_FILE = 'data/sp500_data.parquet'
OUTPUT_FILE = 'data/sp500_factors.parquet'

# ==========================================
# 2. 核心逻辑: 纯 Pandas 手写因子计算
# ==========================================
def compute_factors(df):
    # ----------------------------------------------------
    # A. 趋势指标 (Trend)
    # ----------------------------------------------------
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['Bias_50'] = (df['close'] / df['EMA_50']) - 1

    # ----------------------------------------------------
    # B. 动量指标 (Momentum)
    # ----------------------------------------------------
    # RSI (14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan) # 防止除以0
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['RSI_14'] = df['RSI_14'].fillna(100) # 若无亏损，RSI 为 100
    
    # MACD (12, 26, 9)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # ----------------------------------------------------
    # C. 波动率指标 (Volatility)
    # ----------------------------------------------------
    # ATR (14)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
    df['NATR_14'] = df['ATR_14'] / df['close']

    # ----------------------------------------------------
    # D. 生成标签 (Targets)
    # ----------------------------------------------------
    df['Target_5D'] = df['close'].pct_change(5).shift(-5)
    df['Target_Dir'] = (df['Target_5D'] > 0).astype(int)

    return df

# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"[Error] 找不到输入文件: {INPUT_FILE}，请先运行 data_loader.py")
        exit()

    print("[Extract] 正在读取 Parquet 数据...")
    df = pd.read_parquet(INPUT_FILE)
    
    print("[Transform] 正在计算 500 只股票的技术因子 (CPU 多核运算)...")
    df_reset = df.reset_index()
    
    # 按股票代码分组计算因子
    factors_df = df_reset.groupby('ticker', group_keys=False).apply(compute_factors)
    
    # 清洗无法计算的空值行（前序窗口和未来标签导致的 NaN）
    original_len = len(factors_df)
    factors_df = factors_df.dropna()
    print(f"[Clean] 清除缺失值行: {original_len} -> {len(factors_df)}")
    
    # 恢复索引
    if 'date' in factors_df.columns and 'ticker' in factors_df.columns:
        factors_df = factors_df.set_index(['date', 'ticker']).sort_index()
        
    # 保存至本地
    factors_df.to_parquet(OUTPUT_FILE)
    
    print(f"\n[Load] 因子挖掘完成！已保存至: {OUTPUT_FILE}")
    print(f"[Check] 特征矩阵总列数: {factors_df.shape[1]}")
    print("\n因子展示 (尾部 5 行):")
    print(factors_df[['close', 'RSI_14', 'MACD', 'Target_5D']].tail())