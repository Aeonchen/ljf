# 学生学业预测与预警系统

## 当前第一阶段重构目标

本轮重构只做 4 件事：

1. 拆分配置到 `configs/`。
2. 统一训练端与 Web 端的风险映射逻辑。
3. 将训练流程与绘图流程解耦，避免 `plt.show()` 阻断主流程。
4. 为实验脚本迁移到 `experiments/` 预留结构，并把正式入口说明清楚。

## 当前正式入口

- 回归训练：`run.py`
- 预警训练：`run_warning_optimized.py`
- Web 应用：`app/run_web.py`

## 当前目录结构

```text
student-academic-prediction-system/
├── app/
│   ├── run_web.py
│   └── web_app.py
├── configs/
│   ├── risk.py
│   ├── training.py
│   └── web.py
├── data/
├── experiments/
│   ├── analysis/
│   ├── regression/
│   └── warning/
├── models/
├── reports/
├── src/
│   ├── regression/
│   │   ├── models.py
│   │   ├── pipeline.py
│   │   └── trainer.py
│   ├── shared/
│   │   ├── io.py
│   │   ├── paths.py
│   │   └── plotting.py
│   ├── warning/
│   │   └── labels.py
│   ├── basic_models.py
│   ├── data_preprocessing.py
│   └── utils.py
├── tests/
│   ├── test_plotting.py
│   ├── test_preprocessing.py
│   └── test_risk.py
├── config.py
├── run.py
└── run_warning_optimized.py
```

## 运行方式

### 1. 回归训练

```bash
python run.py
```

### 2. 预警训练

```bash
python run_warning_optimized.py
```

### 3. Web 页面

```bash
cd app
python run_web.py
```

## 第一阶段重构后的规则

### 配置来源

- 训练配置：`configs/training.py`
- Web 资源路径：`configs/web.py`
- 风险规则与建议：`configs/risk.py`
- 兼容导入层：`config.py`

### 风险映射统一来源

- 共享函数：`src/warning/labels.py`
- Web 与训练都必须通过这个模块进行高/中/低风险分类。

### 绘图约束

- 所有训练流程只允许保存图片，不直接调用 `plt.show()`。
- 统一保存工具在 `src/shared/plotting.py`。

## 实验脚本说明

以下脚本视为实验或历史实现，后续应迁入 `experiments/`：

- `run_stage2.py`
- `run_stage3_optimized.py`
- `run_improved_warning.py`
- `binary_classification.py`
- `check_features.py`
- `simple_feature_engineering.py`

在第一阶段结束前，这些脚本仍保留在原位置以确保兼容。
