"""训练相关配置。"""

DATA_PATH = "data/DATA (1).csv"
TARGET_COLUMN = "GRADE"
TEST_SIZE = 0.2
RANDOM_STATE = 42

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

BLS_CONFIG = {
    'mapping_nodes': 100,
    'enhancement_nodes': 100,
    'lambda_value': 0.01,
    'activation': 'sigmoid'
}

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

ENSEMBLE_CONFIG = {
    'use_ensemble': True,
    'ensemble_methods': ['voting', 'stacking'],
    'n_folds': 5
}

EVALUATION_CONFIG = {
    'metrics': ['R2', 'RMSE', 'MAE', 'MAPE'],
    'cross_validation_folds': 5,
    'save_predictions': True
}
