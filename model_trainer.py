import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report
import os

INPUT_FILE = 'data/sp500_factors.parquet'

def train_model():
    print("[1/4] 读取特征数据...")
    df = pd.read_parquet(INPUT_FILE)

    # 提取特征列（必须严格排除价格本身和未来的标签，防止数据泄露）
    exclude_cols = ['Target_5D', 'Target_Dir', 'close', 'high', 'low', 'open', 'volume']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"参与训练的特征数量: {len(feature_cols)} 个")

    print("[2/4] 执行时间序列切分 (Time-Series Split)...")
    df = df.reset_index()
    
    # 训练集：模型学习 2020 年至 2023 年底的历史规律
    train_df = df[df['date'] < '2024-01-01']
    # 测试集：用 2024 年及以后的全新数据对模型进行“模拟实盘”考试
    test_df = df[df['date'] >= '2024-01-01']

    X_train = train_df[feature_cols]
    y_train = train_df['Target_Dir']
    X_test = test_df[feature_cols]
    y_test = test_df['Target_Dir']

    print(f"训练集样本量: {X_train.shape[0]} 行")
    print(f"测试集样本量: {X_test.shape[0]} 行")

    print("[3/4] 启动 LightGBM 训练...")
    # random_state=42 确保每次运行结果完全一致，便于复现和调试
    # n_jobs=-1 指示算法调用你电脑所有的 CPU 核心全速运算
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        n_jobs=-1 
    )

    model.fit(X_train, y_train)

    print("[4/4] 正在执行概率阈值回测 (Probability Threshold Backtest)...")
    
    # 获取预测的"概率"而不是直接的"0或1"分类
    # predict_proba 返回两列：[跌的概率, 涨的概率]，我们提取第二列 [:, 1]
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # 将概率和真实收益率拼成一个表，方便对齐分析
    backtest_df = test_df[['date', 'ticker', 'Target_5D']].copy()
    backtest_df['Prob_Up'] = y_pred_prob

    # 设定严格的置信度阈值 (例如：模型认为上涨概率 > 55% 才买入)
    # 你可以后续调整这个值观察变化
    THRESHOLD = 0.55

    # 生成买入信号：概率大于阈值的标记为 True
    backtest_df['Signal'] = backtest_df['Prob_Up'] > THRESHOLD

    # 开始统计回测指标
    total_trades = backtest_df['Signal'].sum()
    print(f"\n=== 模拟实盘回测结果 (买入阈值: {THRESHOLD}) ===")
    print(f"测试集时间段: 2024-01-01 至今")
    print(f"总发出买入信号次数: {total_trades} 次")

    if total_trades > 0:
        # 选出所有发出买入信号的交易记录
        trades = backtest_df[backtest_df['Signal'] == True]
        
        # 策略胜率：持仓 5 天后真实收益 > 0 的比例
        win_rate = (trades['Target_5D'] > 0).mean()
        # 策略平均单笔收益率
        avg_return = trades['Target_5D'].mean()
        
        print(f"策略胜率 (Win Rate): {win_rate:.2%}")
        print(f"策略单笔平均收益 (Avg 5D Return): {avg_return:.2%}")
        
        # ------------------------------------------------
        # 计算基准线 (Benchmark) 用于对比
        # 基准线代表：如果你在 2024 年每天闭着眼睛随机买入测试集里的股票
        # ------------------------------------------------
        benchmark_win_rate = (backtest_df['Target_5D'] > 0).mean()
        benchmark_avg_return = backtest_df['Target_5D'].mean()
        
        print(f"\n--- 基准线对比 (市场平均水平) ---")
        print(f"大盘随机买入胜率: {benchmark_win_rate:.2%}")
        print(f"大盘随机买入收益: {benchmark_avg_return:.2%}")
        
        # 评估模型超额能力
        excess_return = avg_return - benchmark_avg_return
        print(f"\n模型超额单笔收益 (Alpha): {excess_return:.2%}")
    else:
        print("模型十分保守，在当前阈值下没有发出任何买入信号。建议降低 THRESHOLD 值。")

if __name__ == "__main__":
    train_model()
