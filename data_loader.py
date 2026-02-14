import yfinance as yf
import pandas as pd
import os
import requests  # <--- 新增这一行！

# ==========================================
# 1. 配置区域 (Configuration)
# ==========================================
START_DATE = '2020-01-01'
END_DATE = '2025-01-01'
DATA_PATH = 'data'
PARQUET_FILE = f"{DATA_PATH}/sp500_data.parquet"

# ==========================================
# 2. 获取 S&P 500 名单 (Web Scraping - 修复版)
# ==========================================
def get_sp500_tickers():
    print("🌐 正在从维基百科抓取 S&P 500 成分股名单...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # 🕵️ 关键修复：伪装成浏览器 (User-Agent)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        # 1. 先用 requests 带上“身份证”去下载网页源码
        r = requests.get(url, headers=headers)
        
        # 2. 把源码喂给 pandas 解析
        table = pd.read_html(r.text)
        df_table = table[0]
        
        tickers = df_table['Symbol'].tolist()
        
        # 修正代码格式
        tickers = [t.replace('.', '-') for t in tickers]
        
        print(f"✅ 成功获取 {len(tickers)} 只股票代码。")
        return tickers
        
    except Exception as e:
        print(f"❌ 抓取失败: {e}")
        print("⚠️ 启动备用名单 (Tech Stocks Only)...")
        # 如果还是失败，返回一个备用的核心列表，保证流程能跑通
        return ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'INTC', 'QCOM', 'JPM', 'BAC', 'GS']

# ==========================================
# 3. 核心 ETL 逻辑
# ==========================================
def download_and_clean_data(tickers, start, end):
    print(f"🚀 [Extract] 开始批量下载 {len(tickers)} 只股票数据...")
    print("☕ 这可能需要 1-3 分钟，请耐心等待...")
    
    # 批量下载
    # threads=True: 开启多线程下载，速度起飞
    raw_data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True, threads=True)
    
    print("🧹 [Transform] 正在清洗数据格式 (Wide -> Long)...")
    
    stack_list = []
    
    # 遍历所有股票
    for i, ticker in enumerate(tickers):
        try:
            # 提取单只股票
            df_single = raw_data[ticker].copy()
            
            # 如果全是空值，跳过
            if df_single.dropna(how='all').empty:
                continue
            
            # 打标签
            df_single['Ticker'] = ticker
            df_single = df_single.reset_index()
            
            # 为了节省内存，我们只保留核心列 (有些股票可能没有 Volume)
            # 这一步能让你的 Parquet 文件变小
            cols_to_keep = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
            existing_cols = [c for c in cols_to_keep if c in df_single.columns]
            df_single = df_single[existing_cols]
            
            stack_list.append(df_single)
            
            # 进度条效果 (每处理 50 只打印一次)
            if (i + 1) % 50 == 0:
                print(f"   已处理 {i + 1}/{len(tickers)} 只...")
                
        except KeyError:
            continue

    # 合并
    if stack_list:
        clean_df = pd.concat(stack_list, axis=0)
        
        # 标准化列名
        clean_df.columns = [c.lower() for c in clean_df.columns]
        
        # 设置索引
        if 'date' in clean_df.columns and 'ticker' in clean_df.columns:
            clean_df = clean_df.set_index(['date', 'ticker']).sort_index()
        
        print(f"✅ 清洗完成！最终数据形状: {clean_df.shape}")
        return clean_df
    else:
        print("❌ 没有任何数据被处理")
        return None

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
    # 1. 动态获取名单
    sp500_tickers = get_sp500_tickers()
    
    if sp500_tickers:
        # 2. 下载数据
        df = download_and_clean_data(sp500_tickers, START_DATE, END_DATE)
        
        # 3. 保存
        if df is not None:
            df.to_parquet(PARQUET_FILE)
            file_size = os.path.getsize(PARQUET_FILE) / (1024 * 1024) # 转换成 MB
            print(f"\n💾 [Load] 数据已保存至: {PARQUET_FILE}")
            print(f"📦 文件大小: {file_size:.2f} MB")
            
            # 4. 验证
            print("\n🔍 数据验证 (随机抽查):")
            test_read = pd.read_parquet(PARQUET_FILE)
            print(test_read.sample(5)) # 随机看 5 行