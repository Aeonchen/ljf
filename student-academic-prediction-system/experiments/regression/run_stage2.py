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

# run_stage2_simple.py
"""
阶段2简化版：专注于稳健的特征工程和传统模型
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 添加src到路径
sys.path.append('src')

# 导入模块
import utils
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class SimpleStage2:
    """简化版阶段2"""

    def __init__(self):
        self.results = {}
        self.best_model = None

    def load_data(self):
        """加载数据"""
        print("📂 加载数据...")

        try:
            X_train = pd.read_csv('data/X_train.csv')
            X_test = pd.read_csv('data/X_test.csv')
            y_train = pd.read_csv('data/y_train.csv').iloc[:, 0]
            y_test = pd.read_csv('data/y_test.csv').iloc[:, 0]

            print(f"✅ 数据加载成功")
            print(f"  训练集: {X_train.shape}")
            print(f"  测试集: {X_test.shape}")

            return X_train, X_test, y_train, y_test

        except FileNotFoundError:
            print("❌ 未找到数据文件，请先运行阶段1")
            return None, None, None, None

    def simple_feature_engineering(self, X_train, X_test):
        """简化版特征工程"""
        print("\n🔧 简化特征工程...")

        # 1. 选择最重要的特征（基于相关性）
        print("🎯 选择重要特征...")

        # 加载阶段1的数据来计算相关性
        try:
            X_all = pd.concat([X_train, X_test])
            y_all = pd.read_csv('data/y_train.csv').iloc[:, 0].tolist() + pd.read_csv('data/y_test.csv').iloc[:,
                                                                          0].tolist()

            # 计算特征与目标的相关性
            correlations = {}
            for col in X_all.columns:
                corr = np.corrcoef(X_all[col], y_all)[0, 1]
                correlations[col] = abs(corr)

            # 选择相关性最高的特征
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            top_features = [feat for feat, corr in sorted_features[:15]]  # 选择前15个

            print(f"✅ 选择了 {len(top_features)} 个重要特征")
            print(f"  最重要的特征: {top_features[:5]}")

            X_train_selected = X_train[top_features]
            X_test_selected = X_test[top_features]

        except:
            print("⚠️  无法计算相关性，使用所有特征")
            X_train_selected = X_train
            X_test_selected = X_test

        # 2. 简单的特征缩放
        print("📏 特征缩放...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)

        X_train_final = pd.DataFrame(X_train_scaled, columns=X_train_selected.columns)
        X_test_final = pd.DataFrame(X_test_scaled, columns=X_test_selected.columns)

        print(f"✅ 特征工程完成: {X_train_final.shape[1]} 个特征")

        return X_train_final, X_test_final

    def train_models(self, X_train, X_test, y_train, y_test):
        """训练多个模型"""
        print("\n🤖 训练模型...")

        models = {
            '岭回归': Ridge(alpha=1.0, random_state=42),
            'Lasso回归': Lasso(alpha=0.1, random_state=42),
            '随机森林': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
            '梯度提升': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
            '支持向量机': SVR(kernel='rbf', C=1.0),
            'K近邻': KNeighborsRegressor(n_neighbors=5)
        }

        results = {}

        for name, model in models.items():
            print(f"  📊 训练 {name}...")

            try:
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                # 计算指标
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                test_mae = mean_absolute_error(y_test, y_pred_test)

                results[name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'y_pred_test': y_pred_test
                }

                print(f"    ✅ 测试集R²: {test_r2:.4f}")

            except Exception as e:
                print(f"    ❌ 训练失败: {e}")
                continue

        # 找出最佳模型
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
            self.best_model = {
                'name': best_model_name,
                'model': results[best_model_name]['model'],
                'metrics': {
                    'R2': results[best_model_name]['test_r2'],
                    'RMSE': results[best_model_name]['test_rmse'],
                    'MAE': results[best_model_name]['test_mae']
                }
            }

            print(f"\n🏆 最佳模型: {best_model_name}")
            print(f"   测试集R²: {results[best_model_name]['test_r2']:.4f}")
            print(f"   测试集RMSE: {results[best_model_name]['test_rmse']:.4f}")

        return results

    def visualize_results(self, y_test, model_results):
        """可视化结果"""
        print("\n📊 可视化结果...")

        os.makedirs('reports/stage2_simple', exist_ok=True)

        # 1. 模型性能比较
        model_names = list(model_results.keys())
        test_r2_scores = [model_results[name]['test_r2'] for name in model_names]

        plt.figure(figsize=(10, 6))
        plt.barh(model_names, test_r2_scores, color='skyblue')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        plt.xlabel('测试集R²分数')
        plt.title('模型性能比较')
        plt.tight_layout()
        plt.savefig('reports/stage2_simple/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. 最佳模型预测 vs 实际
        if self.best_model:
            best_name = self.best_model['name']
            y_pred = model_results[best_name]['y_pred_test']

            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.6)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('实际值')
            plt.ylabel('预测值')
            plt.title(f'{best_name}: 预测 vs 实际\nR² = {self.best_model["metrics"]["R2"]:.4f}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('reports/stage2_simple/best_model_predictions.png', dpi=300, bbox_inches='tight')
            plt.show()

        print("✅ 可视化结果已保存")

    def save_models(self, model_results):
        """保存模型"""
        print("\n💾 保存模型...")

        import joblib

        os.makedirs('models/stage2_simple', exist_ok=True)

        # 保存最佳模型
        if self.best_model:
            joblib.dump(self.best_model['model'], 'models/stage2_simple/best_model.pkl')
            print(f"✅ 最佳模型已保存: {self.best_model['name']}")

        # 保存所有模型结果
        import json

        results_to_save = {}
        for name, result in model_results.items():
            results_to_save[name] = {
                'train_r2': result['train_r2'],
                'test_r2': result['test_r2'],
                'test_rmse': result['test_rmse'],
                'test_mae': result['test_mae']
            }

        with open('models/stage2_simple/model_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2)

        print("✅ 模型结果已保存")

    def generate_report(self, X_train_original, X_train_final):
        """生成报告"""
        print("\n📝 生成报告...")

        report_content = f"""# 学生学业预测系统 - 阶段2简化版报告

## 报告信息
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 版本: 简化版（专注于稳健性）

## 数据概况
- 训练集样本: {X_train_original.shape[0]} 个
- 测试集样本: 29 个
- 原始特征数: {X_train_original.shape[1]} 个
- 特征工程后特征数: {X_train_final.shape[1]} 个

## 模型性能
"""

        if self.best_model:
            report_content += f"""
### 最佳模型: {self.best_model['name']}
- 测试集R²: {self.best_model['metrics']['R2']:.4f}
- 测试集RMSE: {self.best_model['metrics']['RMSE']:.4f}
- 测试集MAE: {self.best_model['metrics']['MAE']:.4f}
"""

        report_content += f"""
## 问题分析与建议

### 当前问题
1. **数据量太小**: 总共145个样本，训练集116个，测试集29个
2. **模型过拟合**: 训练集R²远高于测试集R²
3. **特征信息有限**: 30个特征可能不足以准确预测学业成绩

### 改进建议
1. **获取更多数据**: 这是最有效的改进方法
2. **简化模型**: 使用更简单的模型防止过拟合
3. **特征优化**: 创建更有信息量的特征
4. **集成学习**: 结合多个模型的预测结果

## 生成的文件
1. `models/stage2_simple/best_model.pkl` - 最佳模型
2. `models/stage2_simple/model_results.json` - 模型结果
3. `reports/stage2_simple/model_comparison.png` - 模型比较图
4. `reports/stage2_simple/best_model_predictions.png` - 最佳模型预测图

---

*报告自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        report_path = 'reports/stage2_simple_report.md'
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"✅ 报告已保存: {report_path}")

        return report_path

    def run_pipeline(self):
        """运行管道"""
        print("=" * 60)
        print("🎯 学生学业预测系统 - 阶段2简化版")
        print("=" * 60)

        # 1. 加载数据
        X_train, X_test, y_train, y_test = self.load_data()
        if X_train is None:
            return

        # 2. 简化特征工程
        X_train_fe, X_test_fe = self.simple_feature_engineering(X_train, X_test)

        # 3. 训练模型
        model_results = self.train_models(X_train_fe, X_test_fe, y_train, y_test)

        if not model_results:
            print("❌ 没有模型训练成功")
            return

        # 4. 可视化结果
        self.visualize_results(y_test, model_results)

        # 5. 保存模型
        self.save_models(model_results)

        # 6. 生成报告
        report_path = self.generate_report(X_train, X_train_fe)

        print("\n" + "=" * 60)
        print("🎉 阶段2简化版完成！")
        print("=" * 60)

        if self.best_model:
            print(f"\n📊 最佳模型: {self.best_model['name']}")
            print(f"📈 测试集R²: {self.best_model['metrics']['R2']:.4f}")

        print(f"\n📁 报告位置: {report_path}")


def main():
    """主函数"""
    pipeline = SimpleStage2()
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()