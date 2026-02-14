import yfinance as yf
import pandas as pd

# 设置 Pandas 显示选项，利用你的大屏幕优势
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("正在尝试下载 Apple (AAPL) 数据...")

# 1. 下载数据：过去 5 年，日线级别
# auto_adjust=True 会自动处理拆股和分红，这是量化里最重要的“复权”
df = yf.download('AAPL', period='5y', auto_adjust=True)

# 2. 检查数据下载是否成功
if not df.empty:
    print("\n✅ 数据下载成功！")
    print(f"数据形状 (行, 列): {df.shape}")
    print("\n前 5 行数据预览:")
    print(df.head())

    # 3. 简单的计算测试：计算 20 日移动平均线
    df['MA20'] = df['Close'].rolling(window=20).mean()
    print("\n计算 MA20 均线成功，最新一日数据:")
    print(df.tail(1)[['Close', 'MA20']])
else:
    print("❌ 数据下载失败，请检查网络连接 (yfinance 需要访问 Yahoo Finance)。")