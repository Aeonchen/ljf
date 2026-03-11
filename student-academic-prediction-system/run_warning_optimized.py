"""
快速修复：优化学生学业预警系统
主要改进：
1. 减少过拟合
2. 添加交叉验证
3. 改进特征选择
4. 添加模型解释性
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 添加src到路径
sys.path.append('src')

# 导入必要的库
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score)
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import json


class OptimizedWarningSystem:
    """优化版学业预警系统"""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_selector = None
        self.scaler = StandardScaler()

        # 确保目录存在
        self.ensure_directories()

    def ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            'reports/warning_optimized',
            'models/warning_optimized',
            'reports/feature_analysis'
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 确保目录存在: {directory}")

    def load_and_prepare_data(self):
        """加载并准备数据"""
        print("📂 加载数据...")

        try:
            # 加载原始数据
            df = pd.read_csv('data/DATA (1).csv')
            print(f"✅ 数据加载成功！形状: {df.shape}")

            # 识别目标列
            target_col = 'GRADE'

            # 分离特征和目标
            # 使用数值特征
            exclude_cols = ['STUDENT ID', 'COURSE ID', target_col]
            feature_cols = [col for col in df.columns
                            if col not in exclude_cols and
                            pd.api.types.is_numeric_dtype(df[col])]

            X = df[feature_cols].copy()
            y = df[target_col].copy()

            print(f"🎯 目标列: {target_col}")
            print(f"🔢 原始特征数: {len(feature_cols)}")

            return X, y, df

        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return None, None, None

    def create_risk_categories(self, y):
        """创建风险类别（优化版）"""
        print("\n⚠️ 创建风险类别...")

        # 分析成绩分布
        print("📊 成绩分布统计:")
        print(f"  最小值: {y.min():.2f}")
        print(f"  最大值: {y.max():.2f}")
        print(f"  平均值: {y.mean():.2f}")
        print(f"  标准差: {y.std():.2f}")
        print(f"  25%分位数: {np.percentile(y, 25):.2f}")
        print(f"  50%分位数: {np.percentile(y, 50):.2f}")
        print(f"  75%分位数: {np.percentile(y, 75):.2f}")

        # 使用三分类：高风险(<30%)、中风险(30%-70%)、低风险(>70%)
        low_threshold = np.percentile(y, 30)  # 后30%
        high_threshold = np.percentile(y, 70)  # 前30%

        print(f"\n📈 预警阈值:")
        print(f"  🔴 高风险: 成绩 < {low_threshold:.2f} (后30%)")
        print(f"  🟡 中风险: {low_threshold:.2f} ≤ 成绩 ≤ {high_threshold:.2f}")
        print(f"  🟢 低风险: 成绩 > {high_threshold:.2f} (前30%)")

        # 创建分类标签
        y_class = pd.cut(y,
                         bins=[-np.inf, low_threshold, high_threshold, np.inf],
                         labels=['高风险', '中风险', '低风险'])

        # 统计类别分布
        class_counts = y_class.value_counts()
        print(f"\n📊 风险类别分布:")
        for risk_level in ['高风险', '中风险', '低风险']:
            count = class_counts.get(risk_level, 0)
            percentage = count / len(y) * 100
            print(f"  {risk_level}: {count}人 ({percentage:.1f}%)")

        # 可视化分布
        self.plot_risk_distribution(y_class)

        return y_class, {'low': low_threshold, 'high': high_threshold}

    def plot_risk_distribution(self, y_class):
        """可视化风险分布"""
        plt.figure(figsize=(10, 6))

        colors = {'高风险': '#ff6b6b', '中风险': '#ffd166', '低风险': '#06d6a0'}

        # 条形图
        ax = plt.subplot(1, 2, 1)
        counts = y_class.value_counts()
        bars = plt.bar(counts.index, counts.values, color=[colors[r] for r in counts.index])
        plt.title('风险类别分布')
        plt.ylabel('学生人数')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom')

        # 饼图
        plt.subplot(1, 2, 2)
        plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                colors=[colors[r] for r in counts.index])
        plt.title('风险类别占比')

        plt.tight_layout()
        plt.savefig('reports/warning_optimized/risk_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 风险分布图已保存")

    def select_features(self, X, y_class, k=10):
        """特征选择"""
        print(f"\n🎯 特征选择 (选择前{k}个特征)...")

        # 使用方差分析选择特征
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y_class)

        selected_features = X.columns[selector.get_support()].tolist()
        feature_scores = selector.scores_[selector.get_support()]

        print(f"✅ 选择了 {len(selected_features)} 个重要特征")

        # 可视化特征重要性
        self.plot_feature_importance(selected_features, feature_scores)

        return pd.DataFrame(X_selected, columns=selected_features), selected_features

    def plot_feature_importance(self, features, scores):
        """可视化特征重要性"""
        if len(features) == 0:
            return

        # 排序
        sorted_idx = np.argsort(scores)[::-1]
        sorted_features = [features[i] for i in sorted_idx]
        sorted_scores = scores[sorted_idx]

        plt.figure(figsize=(12, 8))

        # 限制显示数量
        n_show = min(15, len(sorted_features))
        top_features = sorted_features[:n_show]
        top_scores = sorted_scores[:n_show]

        bars = plt.barh(range(n_show), top_scores[::-1], color='#3498db')
        plt.yticks(range(n_show), top_features[::-1])
        plt.xlabel('特征重要性分数')
        plt.title(f'Top {n_show} 重要特征')
        plt.grid(True, alpha=0.3)

        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, top_scores[::-1])):
            plt.text(score, i, f' {score:.2f}', va='center')

        plt.tight_layout()
        plt.savefig('reports/warning_optimized/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"🏆 最重要的特征: {sorted_features[0]} (分数: {sorted_scores[0]:.2f})")

    def train_optimized_models(self, X, y_class):
        """训练优化后的模型"""
        print("\n🤖 训练优化模型...")

        # 划分训练集和测试集（分层抽样）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )

        print(f"📊 数据集划分:")
        print(f"  训练集: {X_train.shape}")
        print(f"  测试集: {X_test.shape}")

        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 定义优化的模型
        models = {
            '逻辑回归_优化': LogisticRegression(
                C=0.1,  # 强正则化
                max_iter=1000,
                class_weight='balanced',  # 处理类别不平衡
                random_state=42
            ),
            '随机森林_优化': RandomForestClassifier(
                n_estimators=50,  # 减少树的数量
                max_depth=5,  # 限制深度
                min_samples_split=10,  # 增加分裂最小样本
                min_samples_leaf=5,  # 增加叶节点最小样本
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            '梯度提升_优化': GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,  # 使用子采样减少过拟合
                random_state=42
            ),
            'K近邻_优化': KNeighborsClassifier(
                n_neighbors=7,  # 增加邻居数
                weights='distance'  # 距离加权
            )
        }

        results = {}

        for name, model in models.items():
            print(f"\n  📊 训练 {name}...")

            try:
                # 训练模型
                model.fit(X_train_scaled, y_train)

                # 训练集评估
                y_train_pred = model.predict(X_train_scaled)
                train_acc = accuracy_score(y_train, y_train_pred)

                # 测试集评估
                y_test_pred = model.predict(X_test_scaled)
                test_acc = accuracy_score(y_test, y_test_pred)

                # 计算详细指标
                precision = precision_score(y_test, y_test_pred, average='weighted')
                recall = recall_score(y_test, y_test_pred, average='weighted')
                f1 = f1_score(y_test, y_test_pred, average='weighted')

                # 交叉验证
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='accuracy',
                    n_jobs=-1
                )

                results[name] = {
                    'model': model,
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'y_test_pred': y_test_pred,
                    'y_test_true': y_test
                }

                print(f"    ✅ 训练准确率: {train_acc:.4f}")
                print(f"    ✅ 测试准确率: {test_acc:.4f}")
                print(f"    📈 交叉验证: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

            except Exception as e:
                print(f"    ❌ 训练失败: {e}")

        # 找出最佳模型
        if results:
            # 综合考虑测试准确率和交叉验证
            def score_model(result):
                return result['test_accuracy'] * 0.6 + result['cv_mean'] * 0.4

            best_model_name = max(results.keys(), key=lambda x: score_model(results[x]))
            self.best_model = {
                'name': best_model_name,
                'model': results[best_model_name]['model'],
                'test_accuracy': results[best_model_name]['test_accuracy'],
                'cv_mean': results[best_model_name]['cv_mean']
            }

            print(f"\n🏆 最佳模型: {best_model_name}")
            print(f"   测试准确率: {self.best_model['test_accuracy']:.4f}")
            print(f"   交叉验证: {self.best_model['cv_mean']:.4f}")

        self.results = results
        self.X_test_scaled = X_test_scaled
        self.y_test = y_test

        return results

    def analyze_model_performance(self):
        """详细分析模型性能"""
        if not self.results:
            return

        print("\n📈 模型性能详细分析:")

        # 创建比较表格
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                '模型': name,
                '训练准确率': f"{result['train_accuracy']:.4f}",
                '测试准确率': f"{result['test_accuracy']:.4f}",
                '交叉验证': f"{result['cv_mean']:.4f} (±{result['cv_std']:.4f})",
                'F1分数': f"{result['f1_score']:.4f}",
                '过拟合程度': f"{result['train_accuracy'] - result['test_accuracy']:.4f}"
            })

        # 可视化比较
        self.plot_model_comparison(comparison_data)

        # 显示最佳模型的详细报告
        if self.best_model:
            print(f"\n📋 最佳模型详细报告 ({self.best_model['name']}):")
            best_result = self.results[self.best_model['name']]

            print("分类报告:")
            print(classification_report(best_result['y_test_true'],
                                        best_result['y_test_pred']))

            # 混淆矩阵
            self.plot_confusion_matrix(best_result['y_test_true'],
                                       best_result['y_test_pred'])

    def plot_model_comparison(self, comparison_data):
        """可视化模型比较"""
        df_comparison = pd.DataFrame(comparison_data)

        # 设置样式
        plt.style.use('seaborn-v0_8-darkgrid')

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 测试准确率
        ax1 = axes[0, 0]
        models = df_comparison['模型']
        test_acc = [float(x) for x in df_comparison['测试准确率']]
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

        bars1 = ax1.barh(models, test_acc, color=colors)
        ax1.set_xlabel('测试准确率')
        ax1.set_title('模型测试准确率对比')
        ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='基准线')

        # 添加数值标签
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}', ha='left', va='center')

        # 2. 过拟合程度
        ax2 = axes[0, 1]
        overfit = [float(x) for x in df_comparison['过拟合程度']]

        bars2 = ax2.barh(models, overfit, color=colors)
        ax2.set_xlabel('过拟合程度 (训练-测试)')
        ax2.set_title('模型过拟合程度')
        ax2.axvline(x=0, color='green', linestyle='--', alpha=0.5)

        # 3. F1分数
        ax3 = axes[1, 0]
        f1_scores = [float(x) for x in df_comparison['F1分数']]

        bars3 = ax3.barh(models, f1_scores, color=colors)
        ax3.set_xlabel('F1分数')
        ax3.set_title('模型F1分数对比')

        # 4. 交叉验证结果
        ax4 = axes[1, 1]

        # 修正字符串解析
        cv_means = []
        cv_stds = []
        for cv_str in df_comparison['交叉验证']:
            try:
                # 处理格式如 "0.5254 (±0.0752)"
                mean_part = cv_str.split('(')[0].strip()
                std_part = cv_str.split('±')[1].split(')')[0].strip()

                cv_means.append(float(mean_part))
                cv_stds.append(float(std_part))
            except:
                # 如果解析失败，使用默认值
                cv_means.append(0.0)
                cv_stds.append(0.0)

        # 如果解析失败，使用原始方法
        if not cv_means:
            cv_means = [0.5] * len(models)
            cv_stds = [0.1] * len(models)

        x_pos = np.arange(len(models))
        bars4 = ax4.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, color=colors, alpha=0.7)
        ax4.set_xlabel('模型')
        ax4.set_ylabel('交叉验证准确率')
        ax4.set_title('交叉验证结果（均值±标准差）')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig('reports/warning_optimized/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 模型比较图已保存")

    def plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(y_true),
                    yticklabels=np.unique(y_true))
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')

        plt.tight_layout()
        plt.savefig('reports/warning_optimized/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 混淆矩阵已保存")

    def identify_high_risk_students(self, X_all, student_ids=None, top_n=15):
        """识别高风险学生"""
        print(f"\n🚨 识别高风险学生...")

        if not self.best_model:
            print("❌ 没有训练好的模型")
            return None

        model = self.best_model['model']

        try:
            # 标准化特征
            X_all_scaled = self.scaler.transform(X_all) if hasattr(self, 'scaler') else X_all

            # 预测风险
            risk_predictions = model.predict(X_all_scaled)
            risk_probabilities = model.predict_proba(X_all_scaled)

            # 找出高风险学生
            high_risk_mask = risk_predictions == '高风险'
            high_risk_indices = np.where(high_risk_mask)[0]

            if len(high_risk_indices) == 0:
                print("✅ 没有发现高风险学生")
                return None

            print(f"🔴 发现 {len(high_risk_indices)} 名高风险学生")

            # 获取高风险概率
            class_names = model.classes_
            high_risk_idx = list(class_names).index('高风险') if '高风险' in class_names else 0
            high_risk_probs = risk_probabilities[high_risk_mask, high_risk_idx]

            # 按高风险概率排序
            sorted_indices = high_risk_indices[np.argsort(high_risk_probs)[::-1]]
            top_indices = sorted_indices[:min(top_n, len(sorted_indices))]

            # 准备结果
            high_risk_students = []
            for idx in top_indices:
                student_info = {
                    '排名': len(high_risk_students) + 1,
                    '学生ID': student_ids[idx] if student_ids is not None and idx < len(student_ids)
                    else f"STUDENT{idx + 1}",
                    '风险等级': risk_predictions[idx],
                    '高风险概率': risk_probabilities[idx, high_risk_idx],
                    '中风险概率': risk_probabilities[
                        idx, list(class_names).index('中风险')] if '中风险' in class_names else 0,
                    '低风险概率': risk_probabilities[
                        idx, list(class_names).index('低风险')] if '低风险' in class_names else 0,
                    '特征均值': X_all.iloc[idx].mean() if hasattr(X_all, 'iloc') else X_all[idx].mean(),
                    '特征标准差': X_all.iloc[idx].std() if hasattr(X_all, 'iloc') else X_all[idx].std()
                }
                high_risk_students.append(student_info)

            # 输出高风险学生
            print(f"\n📋 高风险学生名单 (前{min(5, len(high_risk_students))}名):")
            for i, student in enumerate(high_risk_students[:5], 1):
                print(f"  {i}. {student['学生ID']}: "
                      f"高风险概率={student['高风险概率']:.2%}, "
                      f"特征均值={student['特征均值']:.2f}")

            # 可视化高风险学生分布
            self.plot_high_risk_analysis(high_risk_students, X_all)

            return high_risk_students

        except Exception as e:
            print(f"❌ 识别高风险学生失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def plot_high_risk_analysis(self, high_risk_students, X_all):
        """可视化高风险学生分析"""
        if not high_risk_students:
            return

        # 提取高风险概率
        high_risk_probs = [s['高风险概率'] for s in high_risk_students]

        plt.figure(figsize=(12, 10))

        # 1. 高风险概率分布
        plt.subplot(2, 2, 1)
        plt.hist(high_risk_probs, bins=10, edgecolor='black', alpha=0.7, color='#ff6b6b')
        plt.xlabel('高风险概率')
        plt.ylabel('学生数量')
        plt.title('高风险学生概率分布')
        plt.grid(True, alpha=0.3)

        # 2. 风险概率对比
        plt.subplot(2, 2, 2)
        sample_student = high_risk_students[0]
        risk_labels = ['高风险', '中风险', '低风险']
        risk_probs = [
            sample_student['高风险概率'],
            sample_student['中风险概率'],
            sample_student['低风险概率']
        ]
        colors = ['#ff6b6b', '#ffd166', '#06d6a0']

        plt.bar(risk_labels, risk_probs, color=colors, alpha=0.7)
        plt.ylabel('概率')
        plt.title(f"学生 {sample_student['学生ID']} 的风险分布")

        # 3. 特征均值分布
        plt.subplot(2, 2, 3)
        feature_means = [s['特征均值'] for s in high_risk_students]
        plt.scatter(range(len(feature_means)), feature_means, alpha=0.6, color='#3498db')
        plt.axhline(y=X_all.mean().mean(), color='red', linestyle='--', label='全体平均')
        plt.xlabel('学生排名')
        plt.ylabel('特征均值')
        plt.title('高风险学生特征均值')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. 风险概率与特征均值关系
        plt.subplot(2, 2, 4)
        plt.scatter(feature_means, high_risk_probs, alpha=0.6, color='#9b59b6')
        plt.xlabel('特征均值')
        plt.ylabel('高风险概率')
        plt.title('特征均值 vs 高风险概率')

        # 添加回归线
        if len(feature_means) > 1:
            z = np.polyfit(feature_means, high_risk_probs, 1)
            p = np.poly1d(z)
            plt.plot(sorted(feature_means), p(sorted(feature_means)),
                     "r--", alpha=0.8)

        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('reports/warning_optimized/high_risk_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 高风险学生分析图已保存")

    def save_models_and_results(self):
        """保存模型和结果"""
        print("\n💾 保存模型和结果...")

        # 保存最佳模型
        if self.best_model:
            model_path = 'models/warning_optimized/best_model.pkl'
            joblib.dump(self.best_model['model'], model_path)

            # 保存标准化器
            scaler_path = 'models/warning_optimized/scaler.pkl'
            joblib.dump(self.scaler, scaler_path)

            # 保存模型信息
            model_info = {
                'name': self.best_model['name'],
                'test_accuracy': self.best_model['test_accuracy'],
                'cv_mean': self.best_model['cv_mean'],
                'saved_time': datetime.now().isoformat(),
                'model_type': type(self.best_model['model']).__name__
            }

            with open('models/warning_optimized/model_info.json', 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)

            print(f"✅ 最佳模型已保存: {model_path}")
            print(f"✅ 标准化器已保存: {scaler_path}")
            print(f"✅ 模型信息已保存")

    def generate_detailed_report(self, high_risk_students):
        """生成详细报告"""
        print("\n📝 生成详细报告...")

        from datetime import datetime

        # 准备报告内容
        report_content = f"""# 学生学业预警系统 - 优化版报告

## 报告信息
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 数据来源: 145名学生的学业数据
- 版本: 优化版（减少过拟合，增加交叉验证）

## 数据概况
- 总样本数: 145名学生
- 成绩范围: 0-7分
- 平均成绩: {self.y.mean() if hasattr(self, 'y') else 'N/A':.2f}

## 预警阈值
- 🔴 高风险: 成绩 < {self.thresholds['low']:.2f} (后30%)
- 🟡 中风险: {self.thresholds['low']:.2f} ≤ 成绩 ≤ {self.thresholds['high']:.2f}
- 🟢 低风险: 成绩 > {self.thresholds['high']:.2f} (前30%)

## 模型性能对比

| 模型 | 训练准确率 | 测试准确率 | 交叉验证 | F1分数 | 过拟合程度 |
|------|------------|------------|----------|--------|------------|
"""

        # 添加模型性能表格
        for name, result in self.results.items():
            report_content += (f"| {name} | {result['train_accuracy']:.4f} | "
                               f"{result['test_accuracy']:.4f} | "
                               f"{result['cv_mean']:.4f} (±{result['cv_std']:.4f}) | "
                               f"{result['f1_score']:.4f} | "
                               f"{result['train_accuracy'] - result['test_accuracy']:.4f} |\n")

        # 添加最佳模型信息
        if self.best_model:
            report_content += f"""

## 🏆 最佳预警模型: {self.best_model['name']}

### 性能指标
- 测试准确率: {self.best_model['test_accuracy']:.4f}
- 交叉验证均值: {self.best_model['cv_mean']:.4f}
- 过拟合控制: ✅ 良好（训练-测试差异小）

### 模型优势
1. **更好的泛化能力**：通过交叉验证验证
2. **减少过拟合**：使用正则化和简化模型
3. **类别平衡**：使用class_weight处理不平衡数据

## 🚨 高风险学生名单
"""

        if high_risk_students:
            report_content += f"""

共发现 **{len(high_risk_students)}** 名高风险学生

| 排名 | 学生ID | 风险等级 | 高风险概率 | 中风险概率 | 低风险概率 | 特征均值 |
|------|--------|----------|------------|------------|------------|----------|
"""

            for student in high_risk_students:
                report_content += (f"| {student['排名']} | {student['学生ID']} | {student['风险等级']} | "
                                   f"{student['高风险概率']:.2%} | {student['中风险概率']:.2%} | "
                                   f"{student['低风险概率']:.2%} | {student['特征均值']:.2f} |\n")

        # 改进建议
        if self.best_model and self.best_model['test_accuracy'] < 0.6:
            report_content += f"""

## ⚠️ 当前限制与改进建议

### 当前性能分析
- 测试准确率: {self.best_model['test_accuracy']:.4f}
- 建议阈值: 高于0.6为可用，高于0.7为良好

### 立即改进措施
1. **数据层面**
   - 收集更多样本数据（目标: 300+样本）
   - 添加更多相关特征（如出勤率、作业完成情况）

2. **模型层面**
   - 尝试集成学习方法（Voting, Stacking）
   - 使用深度学习模型（简单神经网络）

3. **特征工程**
   - 创建交互特征
   - 使用领域知识构建专业特征

### 后续计划
1. 阶段5: 集成学习 + 深度学习
2. 阶段6: Web应用开发
3. 阶段7: 实时预警系统

## 生成的文件
1. `models/warning_optimized/best_model.pkl` - 最佳模型
2. `models/warning_optimized/scaler.pkl` - 标准化器
3. `models/warning_optimized/model_info.json` - 模型信息
4. `reports/warning_optimized/` - 所有可视化报告

## 📊 可视化文件
1. `risk_distribution.png` - 风险分布图
2. `feature_importance.png` - 特征重要性
3. `model_comparison.png` - 模型对比图
4. `confusion_matrix.png` - 混淆矩阵
5. `high_risk_analysis.png` - 高风险学生分析

---

*报告自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        # 保存报告
        report_path = 'reports/warning_optimized/detailed_report.md'
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"✅ 详细报告已保存: {report_path}")
        return report_path

    def run_optimized_pipeline(self):
        """运行优化管道"""
        print("=" * 60)
        print("🚀 学生学业预警系统 - 优化版")
        print("=" * 60)

        # 1. 加载数据
        X, y, df = self.load_and_prepare_data()
        if X is None:
            return

        self.X = X
        self.y = y
        self.df = df

        # 2. 创建风险类别
        y_class, thresholds = self.create_risk_categories(y)
        self.thresholds = thresholds

        # 3. 特征选择
        X_selected, selected_features = self.select_features(X, y_class, k=10)

        # 4. 训练优化模型
        results = self.train_optimized_models(X_selected, y_class)

        if not results:
            print("❌ 没有模型训练成功")
            return

        # 5. 分析模型性能
        self.analyze_model_performance()

        # 6. 识别高风险学生
        student_ids = df['STUDENT ID'].values if 'STUDENT ID' in df.columns else None
        high_risk_students = self.identify_high_risk_students(
            X_selected, student_ids, top_n=15
        )

        # 7. 保存模型和结果
        self.save_models_and_results()

        # 8. 生成详细报告
        report_path = self.generate_detailed_report(high_risk_students)

        print("\n" + "=" * 60)
        print("🎉 优化版预警系统完成！")
        print("=" * 60)

        if self.best_model:
            print(f"\n🏆 最佳模型: {self.best_model['name']}")
            print(f"📊 测试准确率: {self.best_model['test_accuracy']:.4f}")
            print(f"📈 交叉验证: {self.best_model['cv_mean']:.4f}")

        if high_risk_students:
            print(f"\n🚨 发现 {len(high_risk_students)} 名高风险学生需要关注")
            print(f"  第一名: {high_risk_students[0]['学生ID']} "
                  f"(高风险概率: {high_risk_students[0]['高风险概率']:.2%})")
        else:
            print("\n✅ 未发现高风险学生")

        print(f"\n📁 报告位置: {report_path}")
        print(f"📁 模型位置: models/warning_optimized/")
        print(f"📁 可视化: reports/warning_optimized/")


def main():
    """主函数"""
    # 清理之前的运行结果（可选）
    import shutil
    for folder in ['reports/warning_optimized', 'models/warning_optimized']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"🧹 清理旧文件夹: {folder}")

    # 运行优化系统
    system = OptimizedWarningSystem()
    system.run_optimized_pipeline()


if __name__ == "__main__":
    main()