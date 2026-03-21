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

# run_stage4_warning_working.py
"""
阶段4：学生学业预警系统（工作版）
修复目录不存在的问题，简化代码确保能运行
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# 添加src到路径
sys.path.append('src')


class AcademicWarningSystemWorking:
    """学生学业预警系统（工作版）"""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.thresholds = {}
        self.best_model = None

        # 确保必要的目录存在
        self.ensure_directories()

    def ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            'reports/warning_system',
            'models/warning_system'
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 确保目录存在: {directory}")

    def load_and_prepare_data(self):
        """加载并准备数据（简化版）"""
        print("📂 加载数据...")

        try:
            # 加载原始数据
            df = pd.read_csv('data/DATA (1).csv')

            print(f"📊 数据形状: {df.shape}")

            # 识别目标列
            target_col = 'GRADE'  # 我们知道目标列是GRADE

            # 分离特征和目标
            # 排除非数值列和学生ID、课程ID
            exclude_cols = ['STUDENT ID', 'COURSE ID', target_col]
            feature_cols = [col for col in df.columns if col not in exclude_cols]

            X = df[feature_cols].copy()
            y = df[target_col].copy()

            print(f"🎯 目标列: {target_col}")
            print(f"🔢 特征数: {len(feature_cols)}")

            return X, y, df

        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return None, None, None

    def create_warning_classes(self, y):
        """创建预警类别（简化版）"""
        print("\n⚠️ 创建预警类别...")

        # 查看成绩分布
        print("📊 成绩分布统计:")
        print(f"  最小值: {y.min()}")
        print(f"  最大值: {y.max()}")
        print(f"  平均值: {y.mean():.2f}")
        print(f"  中位数: {y.median():.2f}")

        # 方法：基于成绩分布的百分位数创建预警类别
        low_threshold = np.percentile(y, 30)  # 后30%为高风险
        high_threshold = np.percentile(y, 70)  # 前30%为低风险

        print(f"\n📈 预警阈值:")
        print(f"  高风险: < {low_threshold:.2f}")
        print(f"  中风险: {low_threshold:.2f} - {high_threshold:.2f}")
        print(f"  低风险: > {high_threshold:.2f}")

        # 创建分类标签
        y_class = pd.cut(y,
                         bins=[-np.inf, low_threshold, high_threshold, np.inf],
                         labels=['高风险', '中风险', '低风险'])

        # 统计各类别数量
        class_counts = y_class.value_counts()
        print(f"\n📊 预警类别分布:")
        for risk_level, count in class_counts.items():
            print(f"  {risk_level}: {count}人 ({count / len(y) * 100:.1f}%)")

        self.thresholds = {
            'low': low_threshold,
            'high': high_threshold
        }

        return y_class

    def train_warning_models(self, X, y_class):
        """训练预警模型（简化版）"""
        print("\n🤖 训练预警模型...")

        from sklearn.model_selection import train_test_split

        # 检查数据平衡性
        class_counts = y_class.value_counts()
        print("📊 类别分布:", class_counts.to_dict())

        # 使用分层抽样划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )

        print(f"📊 训练集: {X_train.shape}, 测试集: {X_test.shape}")

        # 创建模型 - 使用简单的模型配置
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier

        models = {
            '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
            '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
            '决策树': DecisionTreeClassifier(random_state=42),
            'K近邻': KNeighborsClassifier(n_neighbors=5)
        }

        results = {}

        for name, model in models.items():
            print(f"  📊 训练 {name}...")

            try:
                model.fit(X_train, y_train)

                # 训练集预测
                y_train_pred = model.predict(X_train)
                train_acc = accuracy_score(y_train, y_train_pred)

                # 测试集预测
                y_test_pred = model.predict(X_test)
                test_acc = accuracy_score(y_test, y_test_pred)

                # 分类报告
                report = classification_report(y_test, y_test_pred, output_dict=True)

                results[name] = {
                    'model': model,
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'y_test_pred': y_test_pred,
                    'y_test_true': y_test,
                    'classification_report': report
                }

                print(f"    ✅ 训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")

            except Exception as e:
                print(f"    ❌ 训练失败: {e}")

        # 找出最佳模型
        if results:
            best_model_name = max(results.keys(),
                                  key=lambda x: results[x]['test_accuracy'])
            self.best_model = {
                'name': best_model_name,
                'model': results[best_model_name]['model'],
                'test_accuracy': results[best_model_name]['test_accuracy']
            }

            print(f"\n🏆 最佳预警模型: {best_model_name}")
            print(f"   测试准确率: {self.best_model['test_accuracy']:.4f}")

        self.results = results
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train

        return results

    def identify_high_risk_students(self, X_all, model, student_ids=None, top_n=10):
        """识别高风险学生（简化版）"""
        print(f"\n🚨 识别高风险学生...")

        # 预测所有学生的风险
        try:
            risk_predictions = model.predict(X_all)
            risk_probabilities = model.predict_proba(X_all)
        except:
            print("⚠️  模型预测失败")
            return None

        # 找出高风险学生
        high_risk_mask = risk_predictions == '高风险'
        high_risk_indices = np.where(high_risk_mask)[0]

        if len(high_risk_indices) == 0:
            print("✅ 没有发现高风险学生")
            return None

        print(f"🔴 发现 {len(high_risk_indices)} 名高风险学生")

        # 按高风险概率排序
        # 确保概率矩阵的形状正确
        if risk_probabilities.shape[1] >= 3:  # 有3个类别
            # 找出'高风险'在类别中的位置
            try:
                class_names = model.classes_
                high_risk_idx = list(class_names).index('高风险')
                high_risk_probs = risk_probabilities[high_risk_mask, high_risk_idx]
            except:
                # 如果找不到，用第一列
                high_risk_probs = risk_probabilities[high_risk_mask, 0]
        else:
            high_risk_probs = risk_probabilities[high_risk_mask, 0]

        sorted_indices = high_risk_indices[np.argsort(high_risk_probs)[::-1]]  # 降序

        # 限制数量
        top_indices = sorted_indices[:min(top_n, len(sorted_indices))]

        # 准备结果
        high_risk_students = []
        for idx in top_indices:
            try:
                student_info = {
                    'index': idx,
                    'student_id': student_ids[idx] if student_ids is not None and idx < len(
                        student_ids) else f"STUDENT{idx + 1}",
                    'risk_level': risk_predictions[idx],
                    'high_risk_probability': risk_probabilities[idx, high_risk_idx] if 'high_risk_idx' in locals() else
                    risk_probabilities[idx, 0],
                    'features_mean': X_all.iloc[idx].mean() if hasattr(X_all, 'iloc') else X_all[idx].mean()
                }
                high_risk_students.append(student_info)
            except:
                continue

        print(f"📋 高风险学生示例:")
        for i, student in enumerate(high_risk_students[:5], 1):
            print(f"  {i}. {student['student_id']}: 高风险概率={student['high_risk_probability']:.2%}")

        return high_risk_students

    def generate_simple_report(self, high_risk_students=None):
        """生成简单报告"""
        print("\n📝 生成报告...")

        from datetime import datetime

        # 准备报告内容
        report_content = f"""# 学生学业预警系统报告

## 报告信息
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 数据来源: 145名学生的学业数据

## 数据概况
- 总样本数: 145名学生
- 成绩范围: 0-7分
- 平均成绩: {self.y.mean() if hasattr(self, 'y') else 'N/A'}

## 预警阈值
- 高风险: 成绩 < {self.thresholds.get('low', 'N/A'):.2f} (后30%)
- 中风险: 成绩 {self.thresholds.get('low', 'N/A'):.2f} - {self.thresholds.get('high', 'N/A'):.2f}
- 低风险: 成绩 > {self.thresholds.get('high', 'N/A'):.2f} (前30%)

## 模型性能
"""

        # 添加模型性能表格
        if self.results:
            report_content += "| 模型 | 训练准确率 | 测试准确率 |\n"
            report_content += "|------|------------|------------|\n"

            for name, result in sorted(self.results.items(),
                                       key=lambda x: x[1]['test_accuracy'],
                                       reverse=True):
                report_content += f"| {name} | {result['train_accuracy']:.4f} | {result['test_accuracy']:.4f} |\n"

        # 添加最佳模型信息
        if self.best_model:
            report_content += f"""

## 最佳预警模型: {self.best_model['name']}
- 测试准确率: {self.best_model['test_accuracy']:.4f}
"""

        # 添加高风险学生名单
        if high_risk_students:
            report_content += f"""

## 高风险学生名单（前{len(high_risk_students)}名）
| 排名 | 学生ID | 风险等级 | 高风险概率 | 特征均值 |
|------|--------|----------|------------|----------|
"""

            for i, student in enumerate(high_risk_students, 1):
                report_content += f"| {i} | {student['student_id']} | {student['risk_level']} | {student['high_risk_probability']:.2%} | {student['features_mean']:.2f} |\n"

        report_content += f"""

## 建议
1. **高风险学生**: 需要重点关注和干预
2. **中风险学生**: 定期检查，提供必要支持
3. **低风险学生**: 鼓励继续保持

## 注意事项
- 本系统基于有限数据开发，结果仅供参考
- 需结合教师观察和其他评估方法
- 高风险学生需要人工核实确认

---
*报告自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        report_path = 'reports/warning_system_report_simple.md'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"✅ 报告已保存: {report_path}")

        return report_path

    def save_models(self):
        """保存模型（简化版）"""
        print("\n💾 保存模型...")

        import joblib
        import json
        from datetime import datetime

        # 保存最佳模型
        if self.best_model:
            try:
                model_path = 'models/warning_system/best_model.pkl'
                joblib.dump(self.best_model['model'], model_path)

                # 保存模型信息
                model_info = {
                    'name': self.best_model['name'],
                    'test_accuracy': self.best_model['test_accuracy'],
                    'thresholds': self.thresholds,
                    'saved_time': datetime.now().isoformat()
                }

                with open('models/warning_system/model_info.json', 'w', encoding='utf-8') as f:
                    json.dump(model_info, f, indent=2, ensure_ascii=False)

                print(f"✅ 最佳预警模型已保存: {self.best_model['name']}")

            except Exception as e:
                print(f"⚠️  保存模型失败: {e}")

    def run_pipeline(self):
        """运行预警系统管道（简化版）"""
        print("=" * 60)
        print("🚨 学生学业预警系统（简化工作版）")
        print("=" * 60)

        # 1. 确保目录存在
        self.ensure_directories()

        # 2. 加载数据
        X, y, df = self.load_and_prepare_data()
        if X is None:
            return

        self.X = X
        self.y = y
        self.df = df

        # 3. 创建预警类别
        y_class = self.create_warning_classes(y)

        # 4. 训练预警模型
        results = self.train_warning_models(X, y_class)

        if not results:
            print("❌ 没有模型训练成功")
            return

        # 5. 识别高风险学生
        student_ids = df['STUDENT ID'].values if 'STUDENT ID' in df.columns else None
        high_risk_students = self.identify_high_risk_students(
            X, self.best_model['model'], student_ids
        )

        # 6. 保存模型
        self.save_models()

        # 7. 生成报告
        report_path = self.generate_simple_report(high_risk_students)

        print("\n" + "=" * 60)
        print("🎉 学生学业预警系统完成！")
        print("=" * 60)

        if self.best_model:
            print(f"\n🏆 最佳模型: {self.best_model['name']}")
            print(f"📊 测试准确率: {self.best_model['test_accuracy']:.4f}")

        if high_risk_students:
            print(f"\n🚨 发现 {len(high_risk_students)} 名高风险学生需要关注")
        else:
            print("\n✅ 未发现高风险学生")

        print(f"\n📁 报告位置: {report_path}")
        print(f"📁 模型位置: models/warning_system/")


def main():
    """主函数"""
    warning_system = AcademicWarningSystemWorking()
    warning_system.run_pipeline()


if __name__ == "__main__":
    main()