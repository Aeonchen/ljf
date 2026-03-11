"""
阶段1：基础模型模块
包含：线性回归、决策树、随机森林等基础模型
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings

warnings.filterwarnings('ignore')


class BasicModels:
    """基础模型类"""

    def __init__(self, random_state=42):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.random_state = random_state

    def initialize_models(self):
        """初始化所有基础模型"""
        print("🤖 初始化基础模型...")

        self.models = {
            '线性回归': LinearRegression(),
            '岭回归': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso回归': Lasso(alpha=0.1, random_state=self.random_state),
            '决策树': DecisionTreeRegressor(max_depth=5, random_state=self.random_state),
            '随机森林': RandomForestRegressor(n_estimators=100, max_depth=5,
                                              random_state=self.random_state, n_jobs=-1),
            '梯度提升': GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                  random_state=self.random_state),
            '支持向量机': SVR(kernel='rbf', C=1.0),
            'K近邻': KNeighborsRegressor(n_neighbors=5)
        }

        print(f"✅ 初始化了 {len(self.models)} 个基础模型")
        return self.models

    def train_models(self, X_train, y_train, X_test, y_test):
        """训练所有模型"""
        print("\n" + "=" * 60)
        print("🏋️  开始训练基础模型")
        print("=" * 60)

        self.results = {}

        for name, model in self.models.items():
            print(f"\n📊 训练 {name}...")

            try:
                # 训练模型
                model.fit(X_train, y_train)

                # 预测
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # 计算指标
                train_metrics = self.calculate_metrics(y_train, y_train_pred, f"{name} (训练集)")
                test_metrics = self.calculate_metrics(y_test, y_test_pred, f"{name} (测试集)")

                # 保存结果
                self.results[name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'y_train_pred': y_train_pred,
                    'y_test_pred': y_test_pred
                }

                print(f"✅ {name} 训练完成")
                print(f"   训练集 R²: {train_metrics['R2']:.4f}")
                print(f"   测试集 R²: {test_metrics['R2']:.4f}")

            except Exception as e:
                print(f"❌ {name} 训练失败: {str(e)}")
                continue

        print("\n✅ 所有模型训练完成！")
        return self.results

    def calculate_metrics(self, y_true, y_pred, model_name=""):
        """计算回归指标"""
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': self.calculate_mape(y_true, y_pred)
        }
        return metrics

    def calculate_mape(self, y_true, y_pred):
        """计算平均绝对百分比误差"""
        # 避免除以0
        mask = y_true != 0
        if np.sum(mask) == 0:
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def compare_models(self):
        """比较所有模型的性能"""
        if not self.results:
            print("❌ 请先训练模型")
            return None

        print("\n" + "=" * 60)
        print("📊 模型性能比较")
        print("=" * 60)

        comparison_data = []

        for name, result in self.results.items():
            train_metrics = result['train_metrics']
            test_metrics = result['test_metrics']

            comparison_data.append({
                '模型': name,
                '训练集_R2': train_metrics['R2'],
                '测试集_R2': test_metrics['R2'],
                '测试集_RMSE': test_metrics['RMSE'],
                '测试集_MAE': test_metrics['MAE'],
                'R2差异': train_metrics['R2'] - test_metrics['R2']  # 过拟合指标
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('测试集_R2', ascending=False)

        print("\n📈 模型性能排序（按测试集R²）:")
        print(comparison_df.to_string(index=False))

        # 找出最佳模型
        best_model_name = comparison_df.iloc[0]['模型']
        self.best_model = {
            'name': best_model_name,
            'model': self.results[best_model_name]['model'],
            'metrics': self.results[best_model_name]['test_metrics']
        }

        print(f"\n🏆 最佳模型: {best_model_name}")
        print(f"   测试集 R²: {self.best_model['metrics']['R2']:.4f}")
        print(f"   测试集 RMSE: {self.best_model['metrics']['RMSE']:.4f}")

        return comparison_df

    def visualize_results(self, y_true, X_test=None, figsize=(15, 10)):
        """可视化模型结果"""
        if not self.results:
            print("❌ 请先训练模型")
            return

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        # 1. 模型R²对比图
        model_names = list(self.results.keys())
        test_r2_scores = [self.results[name]['test_metrics']['R2'] for name in model_names]

        axes[0].barh(model_names, test_r2_scores, color='skyblue')
        axes[0].set_xlabel('R²分数')
        axes[0].set_title('模型性能对比（测试集R²）')
        axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)

        # 2. 预测 vs 实际散点图（最佳模型）
        if self.best_model:
            best_name = self.best_model['name']
            y_pred = self.results[best_name]['y_test_pred']

            axes[1].scatter(y_true, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
            axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                         'r--', lw=2, label='理想线')
            axes[1].set_xlabel('实际值')
            axes[1].set_ylabel('预测值')
            axes[1].set_title(f'{best_name}: 预测 vs 实际')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        # 3. 残差图
        if self.best_model:
            residuals = y_true - y_pred
            axes[2].scatter(y_pred, residuals, alpha=0.6)
            axes[2].axhline(y=0, color='r', linestyle='--')
            axes[2].set_xlabel('预测值')
            axes[2].set_ylabel('残差')
            axes[2].set_title(f'{best_name}: 残差图')
            axes[2].grid(True, alpha=0.3)

        # 4. 误差分布图
        if self.best_model:
            axes[3].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            axes[3].axvline(x=0, color='r', linestyle='--', linewidth=2)
            axes[3].set_xlabel('残差')
            axes[3].set_ylabel('频率')
            axes[3].set_title(f'{best_name}: 误差分布')
            axes[3].grid(True, alpha=0.3)

        # 5. 特征重要性（如果模型有）
        if self.best_model and hasattr(self.best_model['model'], 'feature_importances_'):
            importances = self.best_model['model'].feature_importances_
            feature_names = X_test.columns if X_test is not None else [f'特征{i}' for i in range(len(importances))]

            # 取前10个重要特征
            indices = np.argsort(importances)[-10:]

            axes[4].barh(range(len(indices)), importances[indices], align='center')
            axes[4].set_yticks(range(len(indices)))
            axes[4].set_yticklabels([feature_names[i] for i in indices])
            axes[4].set_xlabel('重要性')
            axes[4].set_title(f'{best_name}: 特征重要性（前10）')

        # 6. 模型对比雷达图（简化为条形图）
        metrics_to_compare = ['R2', 'RMSE', 'MAE']
        n_metrics = len(metrics_to_compare)

        # 取前3个模型
        top_models = list(self.results.keys())[:3]

        for i, model_name in enumerate(top_models):
            metrics = self.results[model_name]['test_metrics']
            values = [metrics[m] for m in metrics_to_compare]

            # 对RMSE和MAE进行归一化（越小越好）
            # 这里简化处理
            x_pos = np.arange(n_metrics)
            width = 0.25
            offset = width * (i - len(top_models) / 2 + 0.5)

            axes[5].bar(x_pos + offset, values, width, label=model_name)

        axes[5].set_xticks(x_pos)
        axes[5].set_xticklabels(metrics_to_compare)
        axes[5].set_ylabel('值')
        axes[5].set_title('前三模型指标对比')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 保存图形
        plt.savefig('reports/model_comparison.png', dpi=300, bbox_inches='tight')
        print("📊 可视化结果已保存到 reports/model_comparison.png")

    def cross_validation(self, X, y, cv=5):
        """交叉验证"""
        print(f"\n🔄 进行 {cv}-折交叉验证...")

        cv_results = {}

        for name, model in self.models.items():
            try:
                # 使用R²作为评分指标
                scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)

                cv_results[name] = {
                    'mean_r2': scores.mean(),
                    'std_r2': scores.std(),
                    'scores': scores
                }

                print(f"✅ {name}: 平均R² = {scores.mean():.4f} (±{scores.std():.4f})")

            except Exception as e:
                print(f"❌ {name} 交叉验证失败: {str(e)}")
                continue

        # 找出交叉验证最佳模型
        if cv_results:
            best_cv_name = max(cv_results.keys(), key=lambda x: cv_results[x]['mean_r2'])
            print(f"\n🏆 交叉验证最佳模型: {best_cv_name}")
            print(f"   平均R²: {cv_results[best_cv_name]['mean_r2']:.4f}")

        return cv_results

    def hyperparameter_tuning(self, X_train, y_train, model_name='随机森林'):
        """超参数调优（示例：随机森林）"""
        print(f"\n🎯 对 {model_name} 进行超参数调优...")

        if model_name not in self.models:
            print(f"❌ 模型 {model_name} 不存在")
            return None

        # 定义参数网格
        if model_name == '随机森林':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_name == '梯度提升':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif model_name == '岭回归':
            param_grid = {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        else:
            print(f"⚠️  暂不支持 {model_name} 的超参数调优")
            return None

        # 网格搜索
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"\n✅ {model_name} 超参数调优完成")
        print(f"   最佳参数: {grid_search.best_params_}")
        print(f"   最佳R²: {grid_search.best_score_:.4f}")

        # 更新模型
        self.models[model_name] = grid_search.best_estimator_

        return grid_search.best_estimator_


def run_basic_model_pipeline(X_train, X_test, y_train, y_test):
    """运行基础模型管道"""
    print("=" * 60)
    print("🤖 开始基础模型训练管道")
    print("=" * 60)

    # 创建模型管理器
    model_manager = BasicModels(random_state=42)

    # 1. 初始化模型
    model_manager.initialize_models()

    # 2. 训练所有模型
    results = model_manager.train_models(X_train, y_train, X_test, y_test)

    # 3. 比较模型
    comparison_df = model_manager.compare_models()

    # 4. 可视化结果
    model_manager.visualize_results(y_test, X_test)

    # 5. 交叉验证
    cv_results = model_manager.cross_validation(X_train, y_train, cv=5)

    # 6. 超参数调优（可选）
    print("\n" + "=" * 60)
    print("⚙️  超参数调优（示例：随机森林）")
    print("=" * 60)

    try:
        best_rf = model_manager.hyperparameter_tuning(X_train, y_train, '随机森林')
    except Exception as e:
        print(f"超参数调优时出错: {str(e)}")

    print("\n" + "=" * 60)
    print("🎉 基础模型训练完成！")
    print("=" * 60)

    return model_manager


if __name__ == "__main__":
    print("🤖 基础模型模块")
    print("使用示例：")
    print("1. 创建BasicModels实例")
    print("2. 调用initialize_models()初始化模型")
    print("3. 调用train_models()训练模型")
    print("4. 调用compare_models()比较性能")
    print("5. 调用visualize_results()可视化")