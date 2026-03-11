student-academic-prediction-system/
├── README.md                           # 项目说明文档
├── requirements.txt                    # 依赖包列表
├── config.py                          # 配置文件
├── data/
│   ├── DATA (1).csv                   # 原始数据
│   ├── processed_data.csv             # 预处理后的数据
│   └── feature_importance.png         # 特征重要性图
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py          # 数据预处理模块
│   ├── feature_engineering.py         # 特征工程模块
│   ├── broad_learning.py              # 宽度学习模型
│   ├── ensemble_warning.py            # 集成学习预警
│   ├── academic_system.py             # 主系统类
│   ├── evaluation.py                  # 评估模块
│   ├── visualization.py               # 可视化模块
│   └── utils.py                       # 工具函数
├── models/
│   ├── predictor_model.pkl            # 训练好的预测模型
│   ├── warning_model.pkl              # 训练好的预警模型
│   └── scaler.pkl                     # 数据标准化器
├── notebooks/
│   ├── 01_data_exploration.ipynb      # 数据探索
│   ├── 02_feature_analysis.ipynb      # 特征分析
│   ├── 03_model_training.ipynb        # 模型训练
│   └── 04_experiment_results.ipynb    # 实验结果
├── app/
│   ├── __init__.py
│   ├── main.py                        # Streamlit主应用
│   ├── pages/
│   │   ├── home.py                    # 首页
│   │   ├── analysis.py                # 分析页面
│   │   ├── prediction.py              # 预测页面
│   │   └── dashboard.py               # 仪表板
│   └── static/
│       ├── style.css                  # 样式文件
│       └── logo.png                   # 图标
├── experiments/
│   ├── model_comparison.py            # 模型对比实验
│   ├── ablation_study.py              # 消融实验
│   └── hyperparameter_tuning.py       # 超参数调优
├── tests/
│   ├── test_preprocessing.py          # 预处理测试
│   ├── test_models.py                 # 模型测试
│   └── test_system.py                 # 系统测试
├── docs/
│   ├── api_docs.md                    # API文档
│   ├── user_guide.md                  # 用户指南
│   └── technical_design.md            # 技术设计
├── reports/
│   ├── model_performance.pdf          # 模型性能报告
│   ├── feature_analysis.pdf           # 特征分析报告
│   └── system_demo.pdf                # 系统演示报告
└── run.py                             # 主运行文件