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
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 添加src到路径
sys.path.append('src')

from sklearn.preprocessing import StandardScaler

from configs.training import DATA_PATH
from src.shared.artifacts import build_manifest, save_manifest
from src.warning.features import plot_feature_importance, select_k_best_features
from src.warning.labels import build_quantile_thresholds, classify_scores
from src.warning.reporting import (
    build_comparison_rows,
    generate_warning_report,
    plot_confusion_matrix,
    plot_high_risk_analysis,
    plot_model_comparison,
    plot_risk_distribution,
    print_best_model_report,
    save_warning_artifacts,
)
from src.warning.trainer import train_warning_models


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
            df = pd.read_csv(DATA_PATH)
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
        """创建风险类别（统一使用共享规则）。"""
        print("\n⚠️ 创建风险类别...")

        print("📊 成绩分布统计:")
        print(f"  最小值: {y.min():.2f}")
        print(f"  最大值: {y.max():.2f}")
        print(f"  平均值: {y.mean():.2f}")
        print(f"  标准差: {y.std():.2f}")
        print(f"  25%分位数: {np.percentile(y, 25):.2f}")
        print(f"  50%分位数: {np.percentile(y, 50):.2f}")
        print(f"  75%分位数: {np.percentile(y, 75):.2f}")

        thresholds = build_quantile_thresholds(y)
        low_threshold = thresholds['low']
        high_threshold = thresholds['high']

        print(f"\n📈 预警阈值:")
        print(f"  🔴 高风险: 成绩 < {low_threshold:.2f} (后30%)")
        print(f"  🟡 中风险: {low_threshold:.2f} ≤ 成绩 < {high_threshold:.2f}")
        print(f"  🟢 低风险: 成绩 ≥ {high_threshold:.2f} (前30%)")

        y_class = classify_scores(y, thresholds)

        class_counts = y_class.value_counts()
        print(f"\n📊 风险类别分布:")
        for risk_level in ['高风险', '中风险', '低风险']:
            count = class_counts.get(risk_level, 0)
            percentage = count / len(y) * 100
            print(f"  {risk_level}: {count}人 ({percentage:.1f}%)")

        self.plot_risk_distribution(y_class)
        return y_class, thresholds

    def plot_risk_distribution(self, y_class):
        """可视化风险分布"""
        plot_risk_distribution(y_class)
        print("📊 风险分布图已保存")

    def select_features(self, X, y_class, k=10):
        """特征选择"""
        print(f"\n🎯 特征选择 (选择前{k}个特征)...")
        X_selected, selected_features, feature_scores = select_k_best_features(X, y_class, k=k)
        self.feature_selector = {
            'selected_features': selected_features,
            'feature_scores': feature_scores,
        }
        print(f"✅ 选择了 {len(selected_features)} 个重要特征")
        self.plot_feature_importance(selected_features, feature_scores)
        return X_selected, selected_features

    def plot_feature_importance(self, features, scores):
        """可视化特征重要性"""
        summary = plot_feature_importance(features, scores)
        if summary is not None:
            print(f"🏆 最重要的特征: {summary['top_feature']} (分数: {summary['top_score']:.2f})")

    def train_optimized_models(self, X, y_class):
        """训练优化后的模型。"""
        print("\n🤖 训练优化模型...")
        training_state = train_warning_models(X, y_class, scaler=self.scaler)
        self.scaler = training_state['scaler']
        self.results = training_state['results']
        self.best_model = training_state['best_model']
        self.X_test_scaled = training_state['X_test_scaled']
        self.y_test = training_state['y_test']

        print("📊 数据集划分:")
        print(f"  训练集: {training_state['X_train_shape']}")
        print(f"  测试集: {training_state['X_test_shape']}")

        for name, result in self.results.items():
            print(f"\n  📊 训练 {name}...")
            print(f"    ✅ 训练准确率: {result['train_accuracy']:.4f}")
            print(f"    ✅ 测试准确率: {result['test_accuracy']:.4f}")
            print(f"    📈 交叉验证: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})")

        print(f"\n🏆 最佳模型: {self.best_model['name']}")
        print(f"   测试准确率: {self.best_model['test_accuracy']:.4f}")
        print(f"   交叉验证: {self.best_model['cv_mean']:.4f}")
        return self.results

    def analyze_model_performance(self):
        """详细分析模型性能"""
        if not self.results:
            return

        print("\n📈 模型性能详细分析:")

        comparison_data = build_comparison_rows(self.results)
        self.plot_model_comparison(comparison_data)

        if self.best_model:
            best_result = self.results[self.best_model['name']]
            print_best_model_report(self.best_model['name'], best_result)
            self.plot_confusion_matrix(best_result['y_test_true'], best_result['y_test_pred'])

    def plot_model_comparison(self, comparison_data):
        """可视化模型比较"""
        plot_model_comparison(comparison_data)
        print("📊 模型比较图已保存")

    def plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        plot_confusion_matrix(y_true, y_pred)
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

            self.plot_high_risk_analysis(high_risk_students, X_all)
            return high_risk_students

        except Exception as e:
            print(f"❌ 识别高风险学生失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def plot_high_risk_analysis(self, high_risk_students, X_all):
        """可视化高风险学生分析"""
        plot_high_risk_analysis(high_risk_students, X_all)
        print("📊 高风险学生分析图已保存")

    def save_models_and_results(self):
        """保存模型和结果"""
        print("\n💾 保存模型和结果...")
        if self.best_model:
            paths = save_warning_artifacts(self.best_model, self.scaler)
            print(f"✅ 最佳模型已保存: {paths['model_path']}")
            print(f"✅ 标准化器已保存: {paths['scaler_path']}")
            print("✅ 模型信息已保存")

    def generate_detailed_report(self, high_risk_students):
        """生成详细报告"""
        print("\n📝 生成详细报告...")
        report_path = generate_warning_report(self, high_risk_students)
        print(f"✅ 详细报告已保存: {report_path}")
        return report_path

    def save_manifest(self, report_path):
        """保存预警训练 manifest。"""
        manifest = build_manifest(
            task='warning',
            model_path='models/warning_optimized/best_model.pkl',
            report_path=report_path,
            scaler_path='models/warning_optimized/scaler.pkl',
            manifest_version='1.0',
            thresholds=self.thresholds,
            data_path=DATA_PATH,
        )
        save_manifest(manifest, 'models/warning_optimized/manifest.json')
        print('✅ 预警 manifest 已保存: models/warning_optimized/manifest.json')

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

        # 9. 保存 manifest
        self.save_manifest(report_path)

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