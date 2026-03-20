"""回归主流程。"""

from datetime import datetime

import pandas as pd

from configs.training import TEST_SIZE
from src import utils
from src.data_preprocessing import DataPreprocessor, check_data_compatibility
from src.regression.trainer import run_regression_training
from src.shared.io import save_dataframe, save_json, save_pickle, save_text



def main():
    print('=' * 60)
    print('🎯 学生学业预测系统 - 阶段1（模块化修复版）')
    print('=' * 60)

    print('\n📁 步骤1: 创建项目目录结构')
    utils.create_project_structure()

    print('\n🔍 步骤2: 检查数据兼容性')
    if not check_data_compatibility():
        print('❌ 数据文件有问题，请检查')
        return

    print('\n🔧 步骤3: 数据预处理')
    preprocessor = DataPreprocessor()
    try:
        X_train, X_test, y_train, y_test = preprocessor.run_pipeline()
        save_dataframe(X_train, 'data/X_train.csv')
        save_dataframe(X_test, 'data/X_test.csv')
        save_dataframe(pd.DataFrame(y_train, columns=[preprocessor.target_column]), 'data/y_train.csv')
        save_dataframe(pd.DataFrame(y_test, columns=[preprocessor.target_column]), 'data/y_test.csv')
        print('💾 预处理数据已保存到 data/ 目录')
    except Exception as exc:
        print(f'❌ 数据预处理失败: {exc}')
        import traceback
        traceback.print_exc()
        return

    analysis_results = None
    print('\n📊 步骤4: 数据可视化分析')
    try:
        X_all = pd.concat([X_train, X_test])
        y_all = pd.concat([pd.Series(y_train), pd.Series(y_test)])
        y_all.name = preprocessor.target_column
        analysis_results = utils.plot_data_analysis(X_all, y_all, preprocessor.target_column)
    except Exception as exc:
        print(f'⚠️  数据可视化时出错: {exc}')

    print('\n🤖 步骤5: 训练基础模型')
    try:
        trainer = run_regression_training(X_train, X_test, y_train, y_test)
        save_pickle(trainer.best_model['model'], 'models/best_model.pkl')
        save_json({
            'name': trainer.best_model['name'],
            'metrics': trainer.best_model['metrics'],
            'saved_time': datetime.now().isoformat(),
            'model_type': type(trainer.best_model['model']).__name__,
        }, 'models/model_info.json')
        print('💾 最佳模型已保存到: models/best_model.pkl')
        print('📄 模型信息已保存到: models/model_info.json')
    except Exception as exc:
        print(f'❌ 模型训练失败: {exc}')
        import traceback
        traceback.print_exc()
        return

    print('\n📝 步骤6: 生成最终报告')
    try:
        report_path = utils.generate_final_report(preprocessor, trainer, analysis_results, test_size=TEST_SIZE)
    except Exception as exc:
        print(f'⚠️  生成报告时出错: {exc}')
        report_path = save_text(
            '# 简单报告\n\n生成时间: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n',
            'reports/simple_report.md',
        )

    print('\n' + '=' * 60)
    print('🎉 阶段1完成！')
    print('=' * 60)
    print('\n📁 生成的文件总结:')
    print('1. 数据文件: data/X_train.csv, data/X_test.csv, data/y_train.csv, data/y_test.csv')
    print('2. 模型文件: models/best_model.pkl, models/model_info.json')
    print('3. 可视化文件: reports/ 目录下的所有PNG文件')
    print(f'4. 最终报告: {report_path}')
    print('\n🎯 下一步:')
    print('阶段2：特征工程 + 宽度学习模型')
