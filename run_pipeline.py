import subprocess
import sys
import time
import os

def run_script(script_name):
    """
    运行指定的 Python 脚本，并捕获输出与异常
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] =========================================")
    print(f"开始执行模块: {script_name}")
    print("=======================================================")
    
    if not os.path.exists(script_name):
        print(f"[错误] 找不到文件: {script_name}")
        return False

    try:
        # 使用当前运行的 Python 解释器执行子脚本
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            text=True,
            capture_output=False # 直接将输出打印到当前终端
        )
        print(f"\n[{time.strftime('%H:%M:%S')}] 模块 {script_name} 执行成功。")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[错误] 模块 {script_name} 执行失败，返回码: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n[系统异常] 运行 {script_name} 时发生未知错误: {str(e)}")
        return False

def main():
    print("启动 DeepAlpha-V1 每日自动化流水线...")
    start_time = time.time()

    # 定义每日需要按顺序执行的任务列表
    # 注：如果你希望每天自动重训模型，可以把 'train_and_save.py' 加在 predictor 前面
    pipeline_steps = [
        'data_loader.py',      # 第一步：更新全市场量价数据
        'factor_miner.py',     # 第二步：生成最新时间序列与横截面因子
        'daily_predictor.py'   # 第三步：加载固化模型，输出明日交易清单
    ]

    for step in pipeline_steps:
        success = run_script(step)
        if not success:
            print(f"\n[流水线中断] 关键任务 {step} 失败，后续步骤已取消。请检查报错信息。")
            sys.exit(1)
            
    # 计算总耗时
    elapsed_time = time.time() - start_time
    print(f"\n=======================================================")
    print(f"DeepAlpha-V1 每日流水线全部执行完毕！总耗时: {elapsed_time:.2f} 秒")
    print("=======================================================")

if __name__ == "__main__":
    main()