from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if Path.cwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# 创建一个快速探索脚本：check_features.py
import pandas as pd
import numpy as np

df = pd.read_csv('data/DATA (1).csv')
print("=== 数据概览 ===")
print(f"形状: {df.shape}")
print(f"列名: {list(df.columns)}")

# 检查目标变量
print("\n=== GRADE（成绩）分布 ===")
print(df['GRADE'].describe())
print(f"最小值: {df['GRADE'].min()}")
print(f"最大值: {df['GRADE'].max()}")
print(f"平均值: {df['GRADE'].mean():.2f}")

# 检查特征
print("\n=== 特征统计 ===")
features = [col for col in df.columns if col not in ['STUDENT ID', 'COURSE ID', 'GRADE']]
print(f"特征数量: {len(features)}")

for i in range(1, 6):  # 只看前5个特征
    if str(i) in features:
        print(f"\n特征{i}:")
        print(f"  最小值: {df[str(i)].min():.2f}")
        print(f"  最大值: {df[str(i)].max():.2f}")
        print(f"  平均值: {df[str(i)].mean():.2f}")
        print(f"  与GRADE相关性: {np.corrcoef(df[str(i)], df['GRADE'])[0,1]:.3f}")