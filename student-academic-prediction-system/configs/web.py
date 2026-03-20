"""Web 应用配置。"""

APP_CONFIG = {
    'title': '学生学业预警系统',
    'description': '基于机器学习的学生学业风险预测系统',
    'version': '1.0.0',
    'author': '学业预警系统团队',
    'contact': 'data@example.com',
}

MODEL_CONFIG = {
    'model_path': 'models/warning_optimized/best_model.pkl',
    'scaler_path': 'models/warning_optimized/scaler.pkl',
    'data_path': 'data/DATA (1).csv',
    'report_path': 'reports/warning_optimized/detailed_report.md',
}
