# src/evaluation.py
"""
评估模块 - 针对小样本数据优化
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """模型评估器 - 针对小样本优化"""

    def __init__(self):
        self.results = {}

    def calculate_metrics(self, y_true, y_pred, model_name=""):
        """计算评估指标（针对小样本优化）"""
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': self.calculate_mape(y_true, y_pred),
            'Explained_Variance': self.calculate_explained_variance(y_true, y_pred)
        }

        if model_name:
            self.results[model_name] = metrics

        return metrics

    def calculate_mape(self, y_true, y_pred):
        """计算平均绝对百分比误差"""
        mask = y_true != 0
        if np.sum(mask) == 0:
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def calculate_explained_variance(self, y_true, y_pred):
        """计算解释方差"""
        return 1 - np.var(y_true - y_pred) / np.var(y_true)

    def cross_validate_robust(self, model, X, y, cv=5, scoring='r2'):
        """鲁棒的交叉验证（针对小样本）"""
        print(f"🔄 进行 {cv}-折交叉验证...")

        # 使用分层K折（对于回归问题，基于目标值分位数分层）
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        scores = []
        fold_details = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            # 训练模型
            model_copy = self._clone_model(model)
            model_copy.fit(X_train_fold, y_train_fold)

            # 预测
            y_pred_fold = model_copy.predict(X_val_fold)

            # 计算分数
            if scoring == 'r2':
                score = r2_score(y_val_fold, y_pred_fold)
            elif scoring == 'neg_mean_squared_error':
                score = -mean_squared_error(y_val_fold, y_pred_fold)
            else:
                score = r2_score(y_val_fold, y_pred_fold)

            scores.append(score)
            fold_details.append({
                'fold': fold + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'score': score
            })

        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores,
            'fold_details': fold_details
        }

    def _clone_model(self, model):
        """克隆模型（简化版）"""
        import copy
        return copy.deepcopy(model)

    def compare_models(self, results_dict):
        """比较多个模型的性能"""
        comparison = []

        for model_name, metrics in results_dict.items():
            comparison.append({
                '模型': model_name,
                'R²': metrics['R2'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'MSE': metrics['MSE'],
                'MAPE': f"{metrics['MAPE']:.2f}%" if not np.isnan(metrics['MAPE']) else 'N/A',
                '解释方差': f"{metrics['Explained_Variance']:.4f}"
            })

        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('R²', ascending=False)

        return comparison_df

    def plot_performance_comparison(self, results_dict, figsize=(14, 10)):
        """绘制模型性能比较图"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 提取数据
        model_names = list(results_dict.keys())
        r2_scores = [results_dict[name]['R2'] for name in model_names]
        rmse_scores = [results_dict[name]['RMSE'] for name in model_names]
        mae_scores = [results_dict[name]['MAE'] for name in model_names]
        explained_var = [results_dict[name]['Explained_Variance'] for name in model_names]

        # 1. R²比较
        axes[0, 0].barh(model_names, r2_scores, color='skyblue')
        axes[0, 0].set_xlabel('R²分数')
        axes[0, 0].set_title('模型R²比较')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='基准线')
        axes[0, 0].legend()

        # 2. RMSE比较
        axes[0, 1].barh(model_names, rmse_scores, color='lightcoral')
        axes[0, 1].set_xlabel('RMSE')
        axes[0, 1].set_title('模型RMSE比较')

        # 3. MAE比较
        axes[1, 0].barh(model_names, mae_scores, color='lightgreen')
        axes[1, 0].set_xlabel('MAE')
        axes[1, 0].set_title('模型MAE比较')

        # 4. 解释方差比较
        axes[1, 1].barh(model_names, explained_var, color='gold')
        axes[1, 1].set_xlabel('解释方差')
        axes[1, 1].set_title('模型解释方差比较')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].axvline(x=1, color='green', linestyle='--', alpha=0.5, label='完美模型')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

        return fig

    def plot_residual_analysis(self, y_true, y_pred, model_name, figsize=(16, 4)):
        """绘制残差分析图"""
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 4, figsize=figsize)

        # 1. 残差散点图
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('预测值')
        axes[0].set_ylabel('残差')
        axes[0].set_title(f'{model_name} - 残差散点图')
        axes[0].grid(True, alpha=0.3)

        # 2. 残差分布
        axes[1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('残差')
        axes[1].set_ylabel('频率')
        axes[1].set_title(f'{model_name} - 残差分布')
        axes[1].grid(True, alpha=0.3)

        # 3. Q-Q图
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title(f'{model_name} - Q-Q图')

        # 4. 残差与特征的关系（简化）
        axes[3].scatter(range(len(residuals)), residuals, alpha=0.6)
        axes[3].axhline(y=0, color='r', linestyle='--')
        axes[3].set_xlabel('样本索引')
        axes[3].set_ylabel('残差')
        axes[3].set_title(f'{model_name} - 残差序列图')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig

    def plot_prediction_vs_actual(self, y_true, y_pred, model_name, figsize=(10, 8)):
        """绘制预测值与实际值对比图"""
        plt.figure(figsize=figsize)

        # 散点图
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5,
                    c='blue', label='预测点')

        # 理想线
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                 'r--', lw=2, label='理想线')

        # 回归线
        coef = np.polyfit(y_true, y_pred, 1)
        poly1d_fn = np.poly1d(coef)
        x_range = np.linspace(y_true.min(), y_true.max(), 100)
        plt.plot(x_range, poly1d_fn(x_range), 'g-', alpha=0.8, label='回归线')

        # 置信区间
        from scipy import stats
        residuals = y_true - y_pred
        std_residuals = np.std(residuals)
        plt.fill_between(x_range,
                         poly1d_fn(x_range) - 1.96 * std_residuals,
                         poly1d_fn(x_range) + 1.96 * std_residuals,
                         alpha=0.2, color='gray', label='95%置信区间')

        # 计算指标
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # 添加文本
        plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}',
                 transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title(f'{model_name} - 预测 vs 实际')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.show()

    def generate_evaluation_report(self, model_name, y_train_true, y_train_pred,
                                   y_test_true, y_test_pred, cv_results=None,
                                   filepath='reports/evaluation_report.md'):
        """生成评估报告"""
        import os

        # 计算指标
        train_metrics = self.calculate_metrics(y_train_true, y_train_pred)
        test_metrics = self.calculate_metrics(y_test_true, y_test_pred)

        report_content = f"""# 模型评估报告

## 模型信息
- 模型名称: {model_name}
- 评估时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 性能指标

### 训练集
| 指标 | 值 | 解释 |
|------|-----|------|
| R²分数 | {train_metrics['R2']:.4f} | 模型解释的方差比例 |
| RMSE | {train_metrics['RMSE']:.4f} | 均方根误差 |
| MAE | {train_metrics['MAE']:.4f} | 平均绝对误差 |
| MSE | {train_metrics['MSE']:.4f} | 均方误差 |
| MAPE | {train_metrics['MAPE']:.2f}% | 平均绝对百分比误差 |
| 解释方差 | {train_metrics['Explained_Variance']:.4f} | 解释方差分数 |

### 测试集
| 指标 | 值 | 解释 |
|------|-----|------|
| R²分数 | {test_metrics['R2']:.4f} | 模型解释的方差比例 |
| RMSE | {test_metrics['RMSE']:.4f} | 均方根误差 |
| MAE | {test_metrics['MAE']:.4f} | 平均绝对误差 |
| MSE | {test_metrics['MSE']:.4f} | 均方误差 |
| MAPE | {test_metrics['MAPE']:.2f}% | 平均绝对百分比误差 |
| 解释方差 | {test_metrics['Explained_Variance']:.4f} | 解释方差分数 |

## 过拟合分析
- 训练集R² - 测试集R²: {train_metrics['R2'] - test_metrics['R2']:.4f}
- 训练集RMSE - 测试集RMSE: {train_metrics['RMSE'] - test_metrics['RMSE']:.4f}
- 过拟合程度: {"严重" if train_metrics['R2'] - test_metrics['R2'] > 0.2 else "中等" if train_metrics['R2'] - test_metrics['R2'] > 0.1 else "轻微"}

"""

        if cv_results:
            report_content += f"""
## 交叉验证结果
- 平均R²: {cv_results['mean']:.4f}
- R²标准差: {cv_results['std']:.4f}
- 各折分数: {', '.join([f'{s:.4f}' for s in cv_results['scores']])}

"""

        # 性能评估
        report_content += f"""
## 性能评估
"""

        if test_metrics['R2'] > 0.7:
            report_content += "✅ **优秀** - 模型性能优秀，可以用于生产环境\n"
        elif test_metrics['R2'] > 0.5:
            report_content += "⚠️  **良好** - 模型性能良好，但还有提升空间\n"
        elif test_metrics['R2'] > 0.3:
            report_content += "⚠️  **一般** - 模型性能一般，需要进一步优化\n"
        elif test_metrics['R2'] > 0:
            report_content += "❌ **较差** - 模型性能较差，但有预测能力\n"
        else:
            report_content += "❌ **很差** - 模型几乎没有预测能力\n"

        # 建议
        report_content += f"""
## 改进建议
"""

        if test_metrics['R2'] < 0.3:
            report_content += """
1. **数据质量**: 检查数据是否存在问题，可能需要更多数据
2. **特征工程**: 创建更有信息量的特征
3. **模型选择**: 尝试不同的模型架构
4. **超参数调优**: 进行更细致的参数优化
"""
        elif train_metrics['R2'] - test_metrics['R2'] > 0.2:
            report_content += """
1. **正则化**: 增加正则化强度
2. **简化模型**: 减少模型复杂度
3. **数据增强**: 增加训练数据或使用数据增强技术
4. **集成学习**: 使用集成方法减少方差
"""
        else:
            report_content += """
1. **继续优化**: 当前模型表现良好，可以继续微调
2. **特征选择**: 尝试不同的特征组合
3. **模型集成**: 集成多个模型可能进一步提升性能
"""

        report_content += f"""
---

*报告自动生成*
"""

        # 保存报告
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"📝 评估报告已保存到: {filepath}")

        return filepath