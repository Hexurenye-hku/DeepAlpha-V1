import pandas as pd
import lightgbm as lgb
import joblib
import os

INPUT_FILE = 'data/sp500_factors.parquet'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'optimal_lgbm.pkl')

def train_and_save():
    print("[1/3] 读取全量历史因子数据...")
    df = pd.read_parquet(INPUT_FILE)
    df = df.reset_index()

    # 严格排除非特征列（防止数据穿越）
    exclude_cols = ['Target_5D', 'Target_Dir', 'close', 'high', 'low', 'open', 'volume', 'date', 'ticker']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print("[2/3] 使用全量数据训练实盘部署模型...")
    X_train = df[feature_cols]
    y_train = df['Target_Dir']

    # 使用回测中已被验证的最佳参数
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("[3/3] 正在序列化保存模型...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    joblib.dump(model, MODEL_PATH)
    print(f"\n操作完成。最终模型已永久保存至: {MODEL_PATH}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"[Error] 找不到输入文件: {INPUT_FILE}")
    else:
        train_and_save()