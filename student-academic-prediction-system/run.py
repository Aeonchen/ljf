"""
学生学业预测系统 - 阶段1（模块化修复版）
修复了数据可视化错误和tabulate依赖问题
使用模块化结构
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加src到路径
sys.path.append('src')

# 从模块导入功能
from data_preprocessing import DataPreprocessor, check_data_compatibility
from basic_models import BasicModels, run_basic_model_pipeline
import utils

# ==================== 配置部分 ====================
DATA_PATH = "data/DATA (1).csv"
TARGET_COLUMN = "GRADE"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ==================== 主函数 ====================
def main():
    """主函数"""
    print("=" * 60)
    print("🎯 学生学业预测系统 - 阶段1（模块化修复版）")
    print("=" * 60)

    # 步骤1：创建项目结构
    print("\n📁 步骤1: 创建项目目录结构")
    utils.create_project_structure()

    # 步骤2：检查数据兼容性
    print("\n🔍 步骤2: 检查数据兼容性")
    if not check_data_compatibility():
        print("❌ 数据文件有问题，请检查")
        return

    # 步骤3：数据预处理
    print("\n🔧 步骤3: 数据预处理")
    preprocessor = DataPreprocessor()

    try:
        X_train, X_test, y_train, y_test = preprocessor.run_pipeline()

        # 保存数据
        X_train.to_csv('data/X_train.csv', index=False)
        X_test.to_csv('data/X_test.csv', index=False)
        pd.DataFrame(y_train, columns=[preprocessor.target_column]).to_csv('data/y_train.csv', index=False)
        pd.DataFrame(y_test, columns=[preprocessor.target_column]).to_csv('data/y_test.csv', index=False)

        print("💾 预处理数据已保存到 data/ 目录")

    except Exception as e:
        print(f"❌ 数据预处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # 步骤4：数据可视化分析
    print("\n📊 步骤4: 数据可视化分析")
    try:
        # 合并训练和测试数据进行整体分析
        X_all = pd.concat([X_train, X_test])
        y_all = pd.concat([pd.Series(y_train), pd.Series(y_test)])
        y_all.name = preprocessor.target_column

        analysis_results = utils.plot_data_analysis(X_all, y_all, preprocessor.target_column)
    except Exception as e:
        print(f"⚠️  数据可视化时出错: {str(e)}")
        analysis_results = None

    # 步骤5：训练基础模型
    print("\n🤖 步骤5: 训练基础模型")
    try:
        # 使用basic_models模块训练
        model_manager = run_basic_model_pipeline(X_train, X_test, y_train, y_test)

        # 保存最佳模型
        import joblib
        os.makedirs('models', exist_ok=True)

        # 保存模型
        model_path = 'models/best_model.pkl'
        joblib.dump(model_manager.best_model['model'], model_path)

        # 保存模型信息
        info_path = 'models/model_info.json'
        model_info = {
            'name': model_manager.best_model['name'],
            'metrics': model_manager.best_model['metrics'],
            'saved_time': datetime.now().isoformat(),
            'model_type': type(model_manager.best_model['model']).__name__
        }

        import json
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)

        print(f"💾 最佳模型已保存到: {model_path}")
        print(f"📄 模型信息已保存到: {info_path}")

    except Exception as e:
        print(f"❌ 模型训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    # 在 run.py 中修改步骤6：

    # 步骤6：生成报告
    print("\n📝 步骤6: 生成最终报告")
    try:
        # 创建模拟的trainer对象以兼容generate_final_report
        class SimpleTrainer:
            def __init__(self, model_manager):
                self.results = model_manager.results if hasattr(model_manager, 'results') else {}
                self.best_model = model_manager.best_model if hasattr(model_manager, 'best_model') else None

        simple_trainer = SimpleTrainer(model_manager)
        report_path = utils.generate_final_report(preprocessor, simple_trainer, analysis_results, test_size=TEST_SIZE)

    except Exception as e:
        print(f"⚠️  生成报告时出错: {str(e)}")
        # 创建简单报告
        with open('reports/simple_report.md', 'w', encoding='utf-8') as f:
            f.write(f"# 简单报告\n\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_path = 'reports/simple_report.md'

    print("\n" + "=" * 60)
    print("🎉 阶段1完成！")
    print("=" * 60)

    print("\n📁 生成的文件总结:")
    print("1. 数据文件: data/X_train.csv, data/X_test.csv, data/y_train.csv, data/y_test.csv")
    print("2. 模型文件: models/best_model.pkl, models/model_info.json")
    print("3. 可视化文件: reports/ 目录下的所有PNG文件")
    print(f"4. 最终报告: {report_path}")

    print("\n🎯 下一步:")
    print("阶段2：特征工程 + 宽度学习模型")

# ==================== 运行程序 ====================
if __name__ == "__main__":
    main()