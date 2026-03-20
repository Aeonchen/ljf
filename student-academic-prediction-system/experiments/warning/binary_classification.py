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

# 创建一个新脚本：binary_classification.py
"""
将三分类问题简化为二分类：
- 高风险（成绩 < 2） vs 非高风险（成绩 >= 2）
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# 加载数据
df = pd.read_csv('data/DATA (1).csv')

# 提取特征和目标
features = [col for col in df.columns if col not in ['STUDENT ID', 'COURSE ID', 'GRADE']]
X = df[features]
y = df['GRADE']

# 创建二分类标签：高风险（<2） vs 非高风险（>=2）
y_binary = (y < 2).astype(int)

print(f"=== 二分类问题 ===")
print(f"高风险（<2分）: {sum(y_binary == 1)} 人 ({sum(y_binary == 1)/len(y)*100:.1f}%)")
print(f"非高风险（>=2分）: {sum(y_binary == 0)} 人 ({sum(y_binary == 0)/len(y)*100:.1f}%)")

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# 训练简单模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n=== 模型性能 ===")
print(f"准确率: {accuracy:.4f}")
print(f"F1分数: {f1:.4f}")

# 特征重要性
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n=== 最重要的5个特征 ===")
print(feature_importance.head())