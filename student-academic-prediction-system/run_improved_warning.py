# 创建新文件：run_improved_warning.py
"""
改进版预警系统：包含特征工程 + 二分类 + 模型优化
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib


def load_and_prepare_data():
    """加载和准备数据"""
    print("📂 加载数据...")

    try:
        # 尝试加载特征工程后的数据
        df = pd.read_csv('data/data_with_new_features.csv')
        print(f"✅ 加载特征工程后数据成功！形状: {df.shape}")
    except:
        # 如果特征工程数据不存在，加载原始数据
        df = pd.read_csv('data/DATA (1).csv')
        print(f"⚠️  使用原始数据，形状: {df.shape}")

    return df


def create_binary_labels(df):
    """创建二分类标签"""
    print("\n🎯 创建二分类标签...")

    # 使用阈值创建二分类
    # 方案1：低于平均值为高风险
    # threshold = df['GRADE'].mean()

    # 方案2：低于2分为高风险（根据之前报告）
    threshold = 2.0

    y_binary = (df['GRADE'] < threshold).astype(int)

    high_risk_count = sum(y_binary == 1)
    high_risk_percent = high_risk_count / len(y_binary) * 100

    print(f"高风险阈值: 成绩 < {threshold:.2f}")
    print(f"高风险学生: {high_risk_count}人 ({high_risk_percent:.1f}%)")
    print(f"非高风险学生: {len(y_binary) - high_risk_count}人 ({100 - high_risk_percent:.1f}%)")

    return y_binary, threshold


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """训练和评估多个模型"""
    print("\n🤖 训练和评估模型...")

    models = {
        '随机森林': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
        '梯度提升': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    }

    results = {}

    for name, model in models.items():
        print(f"  📊 训练 {name}...")

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # 计算指标
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test)

        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

        results[name] = {
            'model': model,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred_test': y_pred_test
        }

        print(f"    ✅ 测试准确率: {test_acc:.4f}, F1: {test_f1:.4f}")
        print(f"    📈 交叉验证: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # 创建投票集成
    print(f"  📊 训练 投票集成...")
    voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='hard'
    )
    voting_clf.fit(X_train, y_train)

    y_pred_test_voting = voting_clf.predict(X_test)
    test_acc_voting = accuracy_score(y_test, y_pred_test_voting)
    test_f1_voting = f1_score(y_test, y_pred_test_voting)

    cv_scores_voting = cross_val_score(voting_clf, X_train, y_train, cv=5, scoring='accuracy')

    results['投票集成'] = {
        'model': voting_clf,
        'train_accuracy': None,  # 投票分类器不方便计算训练集准确率
        'test_accuracy': test_acc_voting,
        'test_f1': test_f1_voting,
        'cv_mean': cv_scores_voting.mean(),
        'cv_std': cv_scores_voting.std(),
        'y_pred_test': y_pred_test_voting
    }

    print(f"    ✅ 测试准确率: {test_acc_voting:.4f}, F1: {test_f1_voting:.4f}")
    print(f"    📈 交叉验证: {cv_scores_voting.mean():.4f} (±{cv_scores_voting.std():.4f})")

    return results


def identify_high_risk_students(df, model, X_test, y_test_pred, student_ids=None):
    """识别高风险学生"""
    print(f"\n🚨 识别高风险学生...")

    # 获取高风险学生的概率
    try:
        # 如果模型支持predict_proba
        probas = model.predict_proba(X_test)
        high_risk_probs = probas[:, 1]  # 第二列是高风险概率
    except:
        # 如果不支持，使用预测结果
        high_risk_probs = y_test_pred

    # 找出预测为高风险的学生
    high_risk_indices = np.where(y_test_pred == 1)[0]

    if len(high_risk_indices) == 0:
        print("✅ 测试集中没有预测为高风险的学生")
        return []

    # 按高风险概率排序
    high_risk_probs_subset = high_risk_probs[high_risk_indices]
    sorted_indices = high_risk_indices[np.argsort(high_risk_probs_subset)[::-1]]

    # 准备结果
    high_risk_students = []
    for idx in sorted_indices[:10]:  # 只显示前10个
        student_info = {
            'index': idx,
            'student_id': student_ids[idx] if student_ids is not None and idx < len(
                student_ids) else f"STUDENT{idx + 1}",
            'actual_grade': df.iloc[idx]['GRADE'] if 'GRADE' in df.columns else 'N/A',
            'high_risk_probability': high_risk_probs[idx] if hasattr(high_risk_probs, '__len__') else 1.0,
            'features_mean': X_test.iloc[idx].mean() if hasattr(X_test, 'iloc') else X_test[idx].mean(),
        }
        high_risk_students.append(student_info)

    print(f"🔴 预测为高风险的学生: {len(high_risk_indices)}人")
    print(f"📋 高风险学生示例（前3名）:")
    for i, student in enumerate(high_risk_students[:3], 1):
        print(
            f"  {i}. {student['student_id']}: 成绩={student['actual_grade']:.2f}, 高风险概率={student['high_risk_probability']:.2%}")

    return high_risk_students


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 改进版学生学业预警系统")
    print("=" * 60)

    # 1. 加载数据
    df = load_and_prepare_data()

    # 2. 准备特征
    if 'GRADE' in df.columns:
        X = df.drop('GRADE', axis=1)
        y, threshold = create_binary_labels(df)
    else:
        print("❌ 数据中没有GRADE列")
        return

    # 3. 获取学生ID（如果存在）
    student_ids = None
    if 'STUDENT ID' in df.columns:
        student_ids = df['STUDENT ID'].values
        X = X.drop('STUDENT ID', axis=1, errors='ignore')

    # 4. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📊 数据集划分:")
    print(f"  训练集: {X_train.shape}")
    print(f"  测试集: {X_test.shape}")

    # 5. 训练和评估模型
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # 6. 找出最佳模型
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_f1'])
    best_model = results[best_model_name]

    print(f"\n🏆 最佳模型: {best_model_name}")
    print(f"   测试准确率: {best_model['test_accuracy']:.4f}")
    print(f"   测试F1分数: {best_model['test_f1']:.4f}")
    print(f"   交叉验证: {best_model['cv_mean']:.4f} (±{best_model['cv_std']:.4f})")

    # 7. 识别高风险学生
    high_risk_students = identify_high_risk_students(
        df, best_model['model'], X_test, best_model['y_pred_test'], student_ids
    )

    # 8. 保存最佳模型
    os.makedirs('models/improved', exist_ok=True)
    model_path = 'models/improved/best_model.pkl'
    joblib.dump(best_model['model'], model_path)
    print(f"\n💾 最佳模型已保存到: {model_path}")

    # 9. 生成简单报告
    report_path = 'reports/improved_report.md'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"""# 改进版预警系统报告

## 最佳模型: {best_model_name}
- 测试准确率: {best_model['test_accuracy']:.4f}
- 测试F1分数: {best_model['test_f1']:.4f}
- 交叉验证: {best_model['cv_mean']:.4f} (±{best_model['cv_std']:.4f})

## 高风险学生（前5名）
""")

        if high_risk_students:
            f.write("| 排名 | 学生ID | 实际成绩 | 高风险概率 |\n")
            f.write("|------|--------|----------|------------|\n")
            for i, student in enumerate(high_risk_students[:5], 1):
                f.write(
                    f"| {i} | {student['student_id']} | {student['actual_grade']:.2f} | {student['high_risk_probability']:.2%} |\n")

    print(f"📝 报告已生成: {report_path}")
    print("\n" + "=" * 60)
    print("🎉 改进版预警系统完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()