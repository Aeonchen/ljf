"""
阶段1：工具函数模块
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 添加这行！

from src.shared.io import load_json, load_pickle, save_json, save_pickle
from src.shared.paths import ensure_directories
from src.shared.plotting import save_figure, safe_close
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def create_project_structure():
    """创建项目目录结构"""
    directories = [
        'data', 'models', 'reports', 'notebooks', 'src', 'experiments', 'logs',
        'configs', 'tests', 'src/regression', 'src/warning', 'src/shared',
        'experiments/regression', 'experiments/warning', 'experiments/analysis'
    ]

    print("📁 创建项目目录结构...")

    for directory in ensure_directories('.', directories):
        print(f"  ✅ 创建目录: {os.path.relpath(directory, '.')}")

    # 创建必要的文件
    files_to_create = {
        'data/README.md': '# 数据目录\n\n存放原始数据和预处理后的数据',
        'models/README.md': '# 模型目录\n\n存放训练好的模型文件',
        'reports/README.md': '# 报告目录\n\n存放分析报告和可视化结果',
        'notebooks/README.md': '# 笔记本目录\n\n存放Jupyter Notebook分析文件',
        'logs/README.md': '# 日志目录\n\n存放运行日志'
    }

    for filepath, content in files_to_create.items():
        if not os.path.exists(filepath):
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ 创建文件: {filepath}")

    print("✅ 项目目录结构创建完成！")


def save_model(model, filepath):
    """保存模型"""
    save_pickle(model, filepath)
    print(f"💾 模型已保存到: {filepath}")
    return True


def load_model(filepath):
    """加载模型"""
    if not os.path.exists(filepath):
        print(f"❌ 模型文件不存在: {filepath}")
        return None

    model = load_pickle(filepath)
    print(f"📂 模型已从 {filepath} 加载")
    return model


def save_results(results, filename='results.json'):
    """保存结果到JSON文件"""
    filepath = os.path.join('reports', filename)

    # 转换numpy数组为列表
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    # 递归转换
    def recursive_convert(obj):
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(item) for item in obj]
        else:
            return convert_numpy(obj)

    results_converted = recursive_convert(results)

    save_json(results_converted, filepath)

    print(f"💾 结果已保存到: {filepath}")
    return filepath


def load_results(filename='results.json'):
    """从JSON文件加载结果"""
    filepath = os.path.join('reports', filename)

    if not os.path.exists(filepath):
        print(f"❌ 结果文件不存在: {filepath}")
        return None

    results = load_json(filepath)

    print(f"📂 结果已从 {filepath} 加载")
    return results


def plot_correlation_matrix(df, figsize=(12, 10), save_path='reports/correlation_matrix.png'):
    """绘制相关系数矩阵"""
    # 确保reports目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=figsize)

    # 计算相关系数（只使用数值列）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("❌ 没有数值型特征，无法绘制相关系数矩阵")
        return None

    corr = df[numeric_cols].corr()

    # 创建热图
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                annot=False, fmt=".2f")

    plt.title('特征相关系数矩阵', fontsize=16)
    plt.tight_layout()

    save_figure(plt.gcf(), save_path)
    safe_close(plt.gcf())
    print(f"📊 相关系数矩阵已保存到: {save_path}")
    return corr


def plot_feature_distributions(df, n_cols=4, figsize=(16, 20), save_path='reports/feature_distributions.png'):
    """绘制特征分布图"""
    # 确保reports目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_features = len(numeric_cols)

    if n_features == 0:
        print("❌ 没有数值型特征")
        return

    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols):
        if idx < len(axes):
            ax = axes[idx]

            # 绘制直方图和密度曲线
            sns.histplot(df[col], kde=True, ax=ax, bins=30)

            # 添加统计信息
            mean_val = df[col].mean()
            median_val = df[col].median()
            std_val = df[col].std()

            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'均值: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', alpha=0.7, label=f'中位数: {median_val:.2f}')

            ax.set_title(f'{col}\n均值: {mean_val:.2f}, 标准差: {std_val:.2f}')
            ax.set_xlabel('')
            ax.legend(fontsize=8)

    # 隐藏多余的子图
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('特征分布分析', fontsize=16, y=1.02)
    plt.tight_layout()

    save_figure(fig, save_path)
    safe_close(fig)
    print(f"📊 特征分布图已保存到: {save_path}")


def generate_report(summary, filename='stage1_report.md'):
    """生成阶段1报告"""
    report_path = os.path.join('reports', filename)

    # 确保reports目录存在
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    report_content = f"""# 阶段1：数据预处理与基础模型报告

## 报告信息
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 阶段: 数据预处理 + 基础模型训练

## 数据概况
- 原始数据文件: {summary.get('data_file', '未知')}
- 总样本数: {summary.get('total_samples', 0)}
- 特征数量: {summary.get('num_features', 0)}
- 目标变量: {summary.get('target_column', '未知')}

## 数据预处理
### 缺失值处理
- 处理前缺失值总数: {summary.get('missing_before', 0)}
- 处理后缺失值总数: {summary.get('missing_after', 0)}

### 数据集划分
- 训练集样本数: {summary.get('train_samples', 0)}
- 测试集样本数: {summary.get('test_samples', 0)}
- 划分比例: {summary.get('test_size', 0.2)*100:.1f}%

### 特征标准化
- 标准化方法: Z-score标准化
- 标准化特征数: {summary.get('scaled_features', 0)}

## 基础模型训练结果

### 模型性能对比
"""

    # 添加模型性能表格
    if 'model_comparison' in summary:
        # 如果有model_comparison数据，将其转换为markdown表格
        try:
            model_df = pd.DataFrame(summary['model_comparison'])
            report_content += "\n" + model_df.to_markdown(index=False)
        except:
            report_content += "\n模型比较数据格式错误"
    else:
        # 如果没有model_comparison数据，使用简化的报告
        report_content += f"""
### 最佳模型信息
- 最佳模型: {summary.get('best_model', {}).get('name', '未知')}
- 测试集R²分数: {summary.get('best_model', {}).get('r2', 0):.4f}
- 测试集RMSE: {summary.get('best_model', {}).get('rmse', 0):.4f}
- 测试集MAE: {summary.get('best_model', {}).get('mae', 0):.4f}

## 目标变量分析
- 目标变量均值: {summary.get('target_mean', 0):.2f}
- 目标变量标准差: {summary.get('target_std', 0):.2f}
- 目标变量范围: [{summary.get('target_min', 0):.2f}, {summary.get('target_max', 0):.2f}]

## 特征相关性
- 与目标最正相关的特征: {summary.get('most_positive_corr', '未知')}
- 与目标最负相关的特征: {summary.get('most_negative_corr', '未知')}
- 平均绝对相关性: {summary.get('avg_abs_correlation', 0):.3f}

## 生成的可视化文件
1. `reports/correlation_matrix.png` - 特征相关系数矩阵
2. `reports/feature_distributions.png` - 特征分布图
3. `reports/model_comparison.png` - 模型性能对比图

## 建议与下一步
1. **数据质量良好**: 缺失值已处理，数据分布合理
2. **模型选择**: 建议使用 {summary.get('best_model', {}).get('name', '随机森林')} 作为基线模型
3. **下一步计划**:
   - 特征工程：创建交互特征、多项式特征
   - 模型优化：超参数调优、集成学习
   - 高级模型：尝试神经网络、宽度学习等

---

*报告自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"📝 报告已生成: {report_path}")
    return report_path


def setup_logging(log_file='logs/stage1.log'):
    """设置日志"""
    import logging

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("日志系统初始化完成")

    return logger


# 在 utils.py 中添加以下函数

# 修改 utils.py 中的 plot_data_analysis 函数：

def plot_data_analysis(X, y, target_name='GRADE'):
    """数据分析和可视化（修复版）"""
    print("\n" + "=" * 60)
    print("📊 数据分析和可视化")
    print("=" * 60)

    # 确保reports目录存在
    os.makedirs('reports', exist_ok=True)

    try:
        # 1. 相关系数矩阵
        print("🔗 计算特征相关系数...")

        # 重置索引，确保索引唯一
        X_reset = X.reset_index(drop=True)
        y_reset = y.reset_index(drop=True) if hasattr(y, 'reset_index') else pd.Series(y).reset_index(drop=True)

        # 合并特征和目标
        data_for_corr = pd.concat([X_reset, y_reset], axis=1)

        # 重命名y列以避免冲突
        y_col_name = 'target' if target_name in X.columns else target_name
        if y_reset.name and y_reset.name in data_for_corr.columns:
            data_for_corr = data_for_corr.rename(columns={y_reset.name: y_col_name})
        else:
            data_for_corr[y_col_name] = y_reset

        # 计算相关系数
        corr_matrix = data_for_corr.corr()

        # 绘制热图
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5},
                    annot=False)

        plt.title('特征相关系数矩阵', fontsize=16)
        plt.tight_layout()
        save_figure(plt.gcf(), 'reports/correlation_matrix.png')
        safe_close(plt.gcf())

        print("📊 相关系数矩阵已保存到 reports/correlation_matrix.png")

        # 2. 特征分布
        print("\n📈 分析特征分布...")

        # 选择前12个特征进行可视化
        n_features_to_plot = min(12, X.shape[1])
        features_to_plot = X.columns[:n_features_to_plot]

        n_cols = 3
        n_rows = (n_features_to_plot + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
        axes = axes.flatten()

        for idx, col in enumerate(features_to_plot):
            ax = axes[idx]
            ax.hist(X[col], bins=20, edgecolor='black', alpha=0.7)
            ax.set_title(f'{col}')
            ax.set_xlabel('值')
            ax.set_ylabel('频数')
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(len(features_to_plot), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('特征分布', fontsize=16, y=1.02)
        plt.tight_layout()
        save_figure(plt.gcf(), 'reports/feature_distributions.png')
        safe_close(plt.gcf())

        print("📊 特征分布图已保存到 reports/feature_distributions.png")

        # 3. 特征与目标的关系
        print("\n🔗 分析特征与目标的关系...")

        # 找出与目标最相关的特征
        target_col_name = 'target' if target_name in X.columns else target_name

        # 计算相关性
        correlations = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                corr = np.corrcoef(X[col], y_reset)[0, 1]
                correlations[col] = corr

        if correlations:
            # 按绝对值排序
            sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            top_features = [feat for feat, _ in sorted_correlations[:5]]

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for idx, feature in enumerate(top_features[:6]):  # 最多显示6个
                if feature in X.columns:
                    ax = axes[idx]
                    ax.scatter(X_reset[feature], y_reset, alpha=0.6)
                    ax.set_xlabel(feature)
                    ax.set_ylabel(target_col_name)
                    ax.set_title(f'{feature} vs {target_col_name}\n相关性: {correlations[feature]:.3f}')
                    ax.grid(True, alpha=0.3)

            plt.suptitle('特征与目标变量关系', fontsize=16, y=1.02)
            plt.tight_layout()
            save_figure(plt.gcf(), 'reports/feature_target_relationships.png')
            safe_close(plt.gcf())

            print("📊 特征与目标关系图已保存到 reports/feature_target_relationships.png")

            # 返回分析结果
            return {
                'correlations': correlations,
                'top_correlated_features': top_features
            }

    except Exception as e:
        print(f"⚠️  数据可视化时出错: {str(e)}")
        import traceback
        traceback.print_exc()

    return None


# 在 utils.py 中修改 generate_final_report 函数：

def generate_final_report(preprocessor, trainer, analysis_results, test_size=0.2, filename='reports/final_report.md'):
    """生成最终报告（修复版，不依赖tabulate）"""
    print("\n" + "=" * 60)
    print("📝 生成最终报告")
    print("=" * 60)

    # 确保reports目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    report_content = f"""# 学生学业预测系统 - 阶段1报告

## 报告信息
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 阶段: 数据预处理 + 基础模型训练

## 数据概况
- 原始数据文件: DATA (1).csv
- 目标变量: {preprocessor.target_column if hasattr(preprocessor, 'target_column') else 'GRADE'}
- 总样本数: {preprocessor.clean_df.shape[0] if hasattr(preprocessor, 'clean_df') else 'N/A'}
- 特征数量: {len(preprocessor.feature_columns) if hasattr(preprocessor, 'feature_columns') else 'N/A'}

## 数据预处理
### 数据清洗
- 处理非数值特征: 自动识别并跳过非数值列
- 缺失值处理: 数值特征用中位数填充
- 异常值处理: 使用IQR方法进行缩尾处理

### 数据集划分
- 训练集比例: {(1 - test_size) * 100:.0f}%
- 测试集比例: {test_size * 100:.0f}%
- 随机种子: {42}

## 模型训练结果

### 模型性能对比
"""

    # 添加模型性能表格
    if trainer.results:
        # 创建自定义表格
        table_header = "| 模型 | R² | RMSE | MAE | MSE |\n"
        table_separator = "|------|------|------|------|------|\n"

        report_content += "\n" + table_header + table_separator

        for name, result in trainer.results.items():
            metrics = result['test_metrics']
            row = f"| {name} | {metrics['R2']:.4f} | {metrics['RMSE']:.4f} | {metrics['MAE']:.4f} | {metrics['MSE']:.4f} |\n"
            report_content += row

    # 添加最佳模型信息
    if trainer.best_model:
        report_content += f"""

### 最佳模型
- 模型名称: {trainer.best_model['name']}
- 测试集R²: {trainer.best_model['metrics']['R2']:.4f}
- 测试集RMSE: {trainer.best_model['metrics']['RMSE']:.4f}
- 测试集MAE: {trainer.best_model['metrics']['MAE']:.4f}

## 特征分析
"""

    # 添加特征相关性信息
    if analysis_results and 'top_correlated_features' in analysis_results:
        report_content += "\n### 与目标最相关的特征（前5）\n"
        for i, feature in enumerate(analysis_results['top_correlated_features'][:5], 1):
            report_content += f"{i}. {feature}\n"

    report_content += f"""
## 生成的文件

### 数据文件
1. `data/X_train.csv` - 训练集特征
2. `data/X_test.csv` - 测试集特征
3. `data/y_train.csv` - 训练集目标
4. `data/y_test.csv` - 测试集目标

### 模型文件
1. `models/best_model.pkl` - 最佳模型
2. `models/model_info.json` - 模型信息

### 可视化报告
1. `reports/target_distribution.png` - 目标变量分布
2. `reports/correlation_matrix.png` - 特征相关系数矩阵
3. `reports/feature_distributions.png` - 特征分布
4. `reports/feature_target_relationships.png` - 特征与目标关系
5. `reports/model_comparison.png` - 模型结果可视化

## 建议与下一步

### 当前成果
✅ 数据预处理完成
✅ 基础模型训练完成
✅ 模型评估完成
✅ 可视化分析完成

### 改进建议
1. **特征工程**: 可以考虑创建交互特征、多项式特征
2. **模型优化**: 对最佳模型进行超参数调优
3. **特征选择**: 使用更先进的特征选择方法
4. **模型集成**: 尝试模型集成方法（如Stacking、Voting）

### 下一步计划
- 阶段2: 特征工程 + 宽度学习模型
- 阶段3: 模型优化 + 集成学习
- 阶段4: 预警系统开发

---

*报告自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"📝 最终报告已保存到: {filename}")
    return filename


if __name__ == "__main__":
    print("🔧 工具函数模块")
    print("可用函数:")
    print("1. create_project_structure() - 创建项目目录")
    print("2. save_model() - 保存模型")
    print("3. load_model() - 加载模型")
    print("4. save_results() - 保存结果")
    print("5. plot_correlation_matrix() - 绘制相关系数矩阵")
    print("6. generate_report() - 生成报告")

    # 创建项目目录
    create_project_structure()
