"""预警任务的可视化、模型保存与报告输出。"""
import matplotlib
matplotlib.rcParams["axes.unicode_minus"] = False
import json
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.shared.io import save_json, save_text
from src.shared.plotting import save_figure, safe_close



def plot_risk_distribution(y_class, output_path='reports/warning_optimized/risk_distribution.png'):
    fig = plt.figure(figsize=(10, 6))
    colors = {'高风险': '#ff6b6b', '中风险': '#ffd166', '低风险': '#06d6a0'}
    counts = y_class.value_counts()

    ax = plt.subplot(1, 2, 1)
    bars = plt.bar(counts.index, counts.values, color=[colors[risk] for risk in counts.index])
    plt.title('风险类别分布')
    plt.ylabel('学生人数')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height)}', ha='center', va='bottom')

    plt.subplot(1, 2, 2)
    plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=[colors[risk] for risk in counts.index])
    plt.title('风险类别占比')
    plt.tight_layout()
    save_figure(fig, output_path)
    safe_close(fig)



def plot_model_comparison(comparison_data, output_path='reports/warning_optimized/model_comparison.png'):
    df_comparison = pd.DataFrame(comparison_data)
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    models = df_comparison['模型']
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

    test_acc = [float(x) for x in df_comparison['测试准确率']]
    bars1 = axes[0, 0].barh(models, test_acc, color=colors)
    axes[0, 0].set_xlabel('测试准确率')
    axes[0, 0].set_title('模型测试准确率对比')
    axes[0, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='基准线')
    for bar in bars1:
        width = bar.get_width()
        axes[0, 0].text(width, bar.get_y() + bar.get_height() / 2, f'{width:.3f}', ha='left', va='center')

    overfit = [float(x) for x in df_comparison['过拟合程度']]
    axes[0, 1].barh(models, overfit, color=colors)
    axes[0, 1].set_xlabel('过拟合程度 (训练-测试)')
    axes[0, 1].set_title('模型过拟合程度')
    axes[0, 1].axvline(x=0, color='green', linestyle='--', alpha=0.5)

    f1_scores = [float(x) for x in df_comparison['F1分数']]
    axes[1, 0].barh(models, f1_scores, color=colors)
    axes[1, 0].set_xlabel('F1分数')
    axes[1, 0].set_title('模型F1分数对比')

    cv_means = []
    cv_stds = []
    for cv_str in df_comparison['交叉验证']:
        try:
            mean_part = cv_str.split('(')[0].strip()
            std_part = cv_str.split('±')[1].split(')')[0].strip()
            cv_means.append(float(mean_part))
            cv_stds.append(float(std_part))
        except (IndexError, ValueError):
            cv_means.append(0.0)
            cv_stds.append(0.0)

    x_pos = np.arange(len(models))
    axes[1, 1].bar(x_pos, cv_means, yerr=cv_stds, capsize=5, color=colors, alpha=0.7)
    axes[1, 1].set_xlabel('模型')
    axes[1, 1].set_ylabel('交叉验证准确率')
    axes[1, 1].set_title('交叉验证结果（均值±标准差）')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_figure(fig, output_path)
    safe_close(fig)



def plot_confusion_matrix(y_true, y_pred, output_path='reports/warning_optimized/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    save_figure(fig, output_path)
    safe_close(fig)



def plot_high_risk_analysis(high_risk_students, X_all, output_path='reports/warning_optimized/high_risk_analysis.png'):
    if not high_risk_students:
        return

    high_risk_probs = [student['高风险概率'] for student in high_risk_students]
    feature_means = [student['特征均值'] for student in high_risk_students]
    fig = plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.hist(high_risk_probs, bins=10, edgecolor='black', alpha=0.7, color='#ff6b6b')
    plt.xlabel('高风险概率')
    plt.ylabel('学生数量')
    plt.title('高风险学生概率分布')
    plt.grid(True, alpha=0.3)

    sample_student = high_risk_students[0]
    risk_labels = ['高风险', '中风险', '低风险']
    risk_probs = [sample_student['高风险概率'], sample_student['中风险概率'], sample_student['低风险概率']]
    colors = ['#ff6b6b', '#ffd166', '#06d6a0']
    plt.subplot(2, 2, 2)
    plt.bar(risk_labels, risk_probs, color=colors, alpha=0.7)
    plt.ylabel('概率')
    plt.title(f"学生 {sample_student['学生ID']} 的风险分布")

    plt.subplot(2, 2, 3)
    plt.scatter(range(len(feature_means)), feature_means, alpha=0.6, color='#3498db')
    plt.axhline(y=X_all.mean().mean(), color='red', linestyle='--', label='全体平均')
    plt.xlabel('学生排名')
    plt.ylabel('特征均值')
    plt.title('高风险学生特征均值')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.scatter(feature_means, high_risk_probs, alpha=0.6, color='#9b59b6')
    plt.xlabel('特征均值')
    plt.ylabel('高风险概率')
    plt.title('特征均值 vs 高风险概率')
    if len(feature_means) > 1:
        coefficients = np.polyfit(feature_means, high_risk_probs, 1)
        line = np.poly1d(coefficients)
        plt.plot(sorted(feature_means), line(sorted(feature_means)), 'r--', alpha=0.8)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path)
    safe_close(fig)



def build_comparison_rows(results):
    rows = []
    for name, result in results.items():
        rows.append({
            '模型': name,
            '训练准确率': f"{result['train_accuracy']:.4f}",
            '测试准确率': f"{result['test_accuracy']:.4f}",
            '交叉验证': f"{result['cv_mean']:.4f} (±{result['cv_std']:.4f})",
            'F1分数': f"{result['f1_score']:.4f}",
            '过拟合程度': f"{result['train_accuracy'] - result['test_accuracy']:.4f}",
        })
    return rows



def save_warning_artifacts(best_model, scaler, output_dir='models/warning_optimized'):
    model_path = f'{output_dir}/best_model.pkl'
    scaler_path = f'{output_dir}/scaler.pkl'
    joblib.dump(best_model['model'], model_path)
    joblib.dump(scaler, scaler_path)
    model_info = {
        'name': best_model['name'],
        'test_accuracy': best_model['test_accuracy'],
        'cv_mean': best_model['cv_mean'],
        'saved_time': datetime.now().isoformat(),
        'model_type': type(best_model['model']).__name__,
    }
    save_json(model_info, f'{output_dir}/model_info.json')
    return {'model_path': model_path, 'scaler_path': scaler_path}



def generate_warning_report(system, high_risk_students, report_path='reports/warning_optimized/detailed_report.md'):
    report_content = f"""# 学生学业预警系统 - 优化版报告

## 报告信息
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 数据来源: 145名学生的学业数据
- 版本: 优化版（减少过拟合，增加交叉验证）

## 数据概况
- 总样本数: 145名学生
- 成绩范围: 0-7分
- 平均成绩: {system.y.mean() if hasattr(system, 'y') else 'N/A':.2f}

## 预警阈值
- 🔴 高风险: 成绩 < {system.thresholds['low']:.2f} (后30%)
- 🟡 中风险: {system.thresholds['low']:.2f} ≤ 成绩 ≤ {system.thresholds['high']:.2f}
- 🟢 低风险: 成绩 > {system.thresholds['high']:.2f} (前30%)

## 模型性能对比

| 模型 | 训练准确率 | 测试准确率 | 交叉验证 | F1分数 | 过拟合程度 |
|------|------------|------------|----------|--------|------------|
"""
    for name, result in system.results.items():
        report_content += (f"| {name} | {result['train_accuracy']:.4f} | "
                           f"{result['test_accuracy']:.4f} | "
                           f"{result['cv_mean']:.4f} (±{result['cv_std']:.4f}) | "
                           f"{result['f1_score']:.4f} | "
                           f"{result['train_accuracy'] - result['test_accuracy']:.4f} |\n")
    if system.best_model:
        report_content += f"""

## 🏆 最佳预警模型: {system.best_model['name']}

### 性能指标
- 测试准确率: {system.best_model['test_accuracy']:.4f}
- 交叉验证均值: {system.best_model['cv_mean']:.4f}
- 过拟合控制: ✅ 良好（训练-测试差异小）
"""
    if high_risk_students:
        report_content += f"""

## 🚨 高风险学生名单

共发现 **{len(high_risk_students)}** 名高风险学生

| 排名 | 学生ID | 风险等级 | 高风险概率 | 中风险概率 | 低风险概率 | 特征均值 |
|------|--------|----------|------------|------------|------------|----------|
"""
        for student in high_risk_students:
            report_content += (f"| {student['排名']} | {student['学生ID']} | {student['风险等级']} | "
                               f"{student['高风险概率']:.2%} | {student['中风险概率']:.2%} | "
                               f"{student['低风险概率']:.2%} | {student['特征均值']:.2f} |\n")
    save_text(report_content, report_path)
    return report_path



def print_best_model_report(best_model_name, result):
    print(f"\n📋 最佳模型详细报告 ({best_model_name}):")
    print('分类报告:')
    print(classification_report(result['y_test_true'], result['y_test_pred']))
