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

# 创建新脚本：simple_feature_engineering.py
import pandas as pd
import numpy as np


def create_simple_features(df):
    """
    创建简单的组合特征
    """
    # 找出数值特征（排除ID和GRADE）
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [f for f in numeric_features if f not in ['STUDENT ID', 'COURSE ID', 'GRADE']]

    print(f"原始特征数: {len(numeric_features)}")

    # 创建新DataFrame
    new_df = pd.DataFrame()
    new_df['GRADE'] = df['GRADE']  # 保留目标变量

    # 1. 基础特征（直接复制）
    for feat in numeric_features:
        new_df[f'f_{feat}'] = df[feat]

    # 2. 统计特征
    if len(numeric_features) > 0:
        # 所有特征的平均值
        new_df['feature_mean'] = df[numeric_features].mean(axis=1)
        # 所有特征的标准差（衡量波动性）
        new_df['feature_std'] = df[numeric_features].std(axis=1)
        # 最大值
        new_df['feature_max'] = df[numeric_features].max(axis=1)
        # 最小值
        new_df['feature_min'] = df[numeric_features].min(axis=1)
        # 极差（最大值-最小值）
        new_df['feature_range'] = new_df['feature_max'] - new_df['feature_min']

    # 3. 简单的交互特征（前5个最重要特征的组合）
    # 先计算相关性，找出最重要的特征
    correlations = {}
    for feat in numeric_features:
        corr = np.corrcoef(df[feat], df['GRADE'])[0, 1]
        correlations[feat] = abs(corr)

    # 选出相关性最高的5个特征
    if len(correlations) >= 5:
        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5]
        top_feature_names = [f[0] for f in top_features]

        print(f"\n与GRADE最相关的5个特征:")
        for feat, corr in top_features:
            print(f"  {feat}: {corr:.3f}")

        # 创建交互特征
        for i in range(len(top_feature_names)):
            for j in range(i + 1, len(top_feature_names)):
                feat1 = top_feature_names[i]
                feat2 = top_feature_names[j]
                new_df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]

    print(f"\n特征工程完成:")
    print(f"原始特征数: {len(numeric_features)}")
    print(f"新特征总数: {len(new_df.columns) - 1}")  # 减去GRADE列

    return new_df


# 主程序
if __name__ == "__main__":
    # 加载数据
    df = pd.read_csv('data/DATA (1).csv')

    # 创建特征
    df_new = create_simple_features(df)

    # 保存新数据
    df_new.to_csv('data/data_with_new_features.csv', index=False)
    print(f"\n新数据已保存到: data/data_with_new_features.csv")
    print(f"新数据形状: {df_new.shape}")