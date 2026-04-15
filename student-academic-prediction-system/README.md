# 学生学业预测与预警系统（精简版）

这是一个可直接运行的“成绩预测 + 学业预警 + Web 展示”项目。

## 项目目标

- 用回归模型预测学生成绩（`GRADE`）
- 将预测结果映射为三类风险（高/中/低）
- 输出预警图表与报告，并在 Web 端展示

## 目录结构

```text
student-academic-prediction-system/
├── app/                     # Web 启动与页面逻辑
├── configs/                 # 训练、风险、Web 配置
├── data/                    # 原始数据与训练/测试数据
├── models/                  # 模型与 manifest
├── reports/                 # 图表、论文文档
├── src/                     # 核心训练/预警/绘图代码
├── tests/                   # 关键测试
├── run.py                   # 回归训练入口
├── run_warning.py           # 预警训练入口（推荐）
└── config.py                # 兼容配置导入
```

> 已移除实验性目录与历史冗余脚本（如 `experiments/`、`notebooks/` 等），保留核心可交付代码。

## 运行方式

### 1) 回归训练

```bash
python run.py
```

### 2) 预警训练

```bash
python run_warning.py
```

### 3) Web 页面

```bash
cd app
python run_web.py
```

## 关键说明

- 统一风险规则：`configs/risk.py`
- 统一绘图保存：`src/shared/plotting.py`
- 预警模型产物：`models/warning_optimized/manifest.json`
- 回归模型产物：`models/regression_manifest.json`
