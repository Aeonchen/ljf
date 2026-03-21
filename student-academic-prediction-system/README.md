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
│   │   ├── artifacts.py
│   │   ├── io.py
│   │   ├── paths.py
│   │   └── plotting.py
│   ├── web/
│   │   └── view_models.py
│   ├── warning/
│   │   ├── features.py
│   │   ├── labels.py
│   │   ├── pipeline.py
│   │   ├── reporting.py
│   │   └── trainer.py
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
python run_warning.py
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

- 风险分类共享函数：`src/warning/labels.py`
- 预警训练编排：`src/warning/pipeline.py`
- 预警特征选择/训练/报告：`src/warning/features.py`、`src/warning/trainer.py`、`src/warning/reporting.py`

### 绘图约束

- 所有训练流程只允许保存图片，不直接调用 `plt.show()`。
- 统一保存工具在 `src/shared/plotting.py`。

### 产物清单（Manifest）

- 预警训练会输出 `models/warning_optimized/manifest.json`。
- 回归训练会输出 `models/regression_manifest.json`。
- Web 会优先读取 manifest，再回退到 `configs/web.py` 中的默认路径。
- `app/web_app.py` 中的资源解析、仪表板摘要、图表所需展示数据、数据概览、单学生预测表单/结果以及学生管理搜索结果拼装已逐步抽到 `src/web/view_models.py`。

## 实验脚本说明

实验脚本已经按职责收敛到 `experiments/` 目录：

- `experiments/regression/run_stage2.py`
- `experiments/warning/run_stage3_optimized.py`
- `experiments/warning/run_improved_warning.py`
- `experiments/warning/binary_classification.py`
- `experiments/analysis/check_features.py`
- `experiments/analysis/simple_feature_engineering.py`

为了兼容旧命令，仓库根目录仍保留同名薄包装脚本，它们会转发到 `experiments/` 中的真实实现。
