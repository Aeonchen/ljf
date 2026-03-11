# config.py
"""
项目配置文件 - 完整版
"""

# 数据配置
DATA_PATH = "data/DATA (1).csv"
TARGET_COLUMN = "GRADE"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 特征名称映射
FEATURE_NAMES = {
    '1': '学生年龄',
    '2': '性别',
    '3': '毕业高中类型',
    '4': '奖学金类型',
    '5': '附加工作',
    '6': '定期的艺术或体育活动',
    '7': '有伴侣',
    '8': '总薪水',
    '9': '前往大学的交通',
    '10': '塞浦路斯住宿类型',
    '11': '母亲教育',
    '12': '父亲教育',
    '13': '兄弟姐妹人数',
    '14': '父母状态',
    '15': '母亲职业',
    '16': '父亲职业',
    '17': '每周学习时数',
    '18': '非科学书籍阅读频率',
    '19': '科学书籍阅读频率',
    '20': '参加研讨会/会议',
    '21': '项目对成功的影响',
    '22': '上课次数',
    '23': '期中考试准备1',
    '24': '期中考试准备2',
    '25': '课堂笔记',
    '26': '课堂听力',
    '27': '讨论提升兴趣',
    '28': '翻转教室',
    '29': '上学期累计平均绩点',
    '30': '毕业预期累计平均绩点'
}

# 宽度学习配置
BLS_CONFIG = {
    'mapping_nodes': 100,
    'enhancement_nodes': 100,
    'lambda_value': 0.01,
    'activation': 'sigmoid'
}

# 特征工程配置
FEATURE_ENGINEERING = {
    'polynomial_degree': 2,
    'interaction_features': True,
    'top_k_features': 15,
    'pca_components': 10,
    'feature_scaling': True,
    'create_cluster_features': True,
    'n_clusters': 3,
    'create_statistical_features': True
}

# 模型集成配置
ENSEMBLE_CONFIG = {
    'use_ensemble': True,
    'ensemble_methods': ['voting', 'stacking'],
    'n_folds': 5
}

# 评估配置
EVALUATION_CONFIG = {
    'metrics': ['R2', 'RMSE', 'MAE', 'MAPE'],
    'cross_validation_folds': 5,
    'save_predictions': True
}
"""
Web应用配置文件
"""

# 应用配置
APP_CONFIG = {
    'title': '学生学业预警系统',
    'description': '基于机器学习的学生学业风险预测系统',
    'version': '1.0.0',
    'author': '学业预警系统团队',
    'contact': 'data@example.com',
}

# 模型配置
MODEL_CONFIG = {
    'model_path': 'models/warning_optimized/best_model.pkl',
    'scaler_path': 'models/warning_optimized/scaler.pkl',
    'data_path': 'data/DATA (1).csv',
    'report_path': 'reports/warning_optimized/detailed_report.md',
}

# 风险阈值配置
RISK_THRESHOLDS = {
    'high_risk': 2.0,      # 成绩 < 2.0 为高风险
    'medium_risk': 5.0,    # 成绩 < 5.0 为中风险
    'low_risk': 5.0,       # 成绩 >= 5.0 为低风险
}

# 重要特征列表（根据分析结果）
IMPORTANT_FEATURES = [
    'feature_29',  # 最重要的特征
    'feature_14',
    'feature_28',
    'feature_12',
    'feature_11',
    'feature_9',
    'feature_5',
    'feature_3',
    'feature_2',
    'feature_1',
]

# 建议内容
RECOMMENDATIONS = {
    'high_risk': [
        '立即安排一对一辅导',
        '联系家长沟通情况',
        '提供额外学习资源',
        '每周检查学习进展',
    ],
    'medium_risk': [
        '定期检查作业完成情况',
        '提供学习技巧指导',
        '鼓励参与小组学习',
        '每月评估学习进步',
    ],
    'low_risk': [
        '给予表扬和鼓励',
        '提供挑战性任务',
        '鼓励帮助其他同学',
        '保持当前学习节奏',
    ],
}