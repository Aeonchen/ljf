"""Streamlit 页面可复用的资源加载与视图模型。"""

import joblib
import pandas as pd

from src.shared.artifacts import load_manifest
from src.warning.labels import build_fixed_thresholds, summarize_risk_levels



def resolve_resource_paths(project_root, model_config):
    manifest_path = project_root / model_config['manifest_path']
    manifest = load_manifest(manifest_path)
    if manifest is not None:
        return {
            'manifest': manifest,
            'model_path': project_root / manifest.get('model_path', model_config['model_path']),
            'scaler_path': project_root / manifest.get('scaler_path', model_config['scaler_path']),
            'data_path': project_root / manifest.get('data_path', model_config['data_path']),
            'report_path': project_root / manifest.get('report_path', model_config['report_path']),
        }
    return {
        'manifest': None,
        'model_path': project_root / model_config['model_path'],
        'scaler_path': project_root / model_config['scaler_path'],
        'data_path': project_root / model_config['data_path'],
        'report_path': project_root / model_config['report_path'],
    }



def load_app_resources(project_root, model_config):
    resolved_paths = resolve_resource_paths(project_root, model_config)
    resources = {'manifest': resolved_paths['manifest'], '_status': []}

    model_path = resolved_paths['model_path']
    scaler_path = resolved_paths['scaler_path']
    data_path = resolved_paths['data_path']
    report_path = resolved_paths['report_path']

    if model_path.exists():
        resources['model'] = joblib.load(model_path)
        resources['_status'].append(('success', f"✅ 模型加载成功: {type(resources['model']).__name__}"))
    else:
        resources['_status'].append(('warning', '⚠️ 模型文件不存在，请先运行训练脚本'))

    if scaler_path.exists():
        resources['scaler'] = joblib.load(scaler_path)
        resources['_status'].append(('success', '✅ 标准化器加载成功'))

    if data_path.exists():
        resources['data'] = pd.read_csv(data_path)
        resources['_status'].append(('success', f"✅ 数据加载成功: {resources['data'].shape[0]} 个样本"))

    if report_path.exists():
        resources['report'] = report_path.read_text(encoding='utf-8')
        resources['_status'].append(('success', '✅ 报告加载成功'))

    resources['_paths'] = resolved_paths
    return resources



def build_dashboard_summary(df, grade_column='GRADE'):
    thresholds = build_fixed_thresholds()
    risk_labels, risk_counts = summarize_risk_levels(df[grade_column], thresholds)
    return {
        'avg_grade': float(df[grade_column].mean()),
        'risk_labels': risk_labels,
        'risk_counts': risk_counts,
        'high_risk_count': int(risk_counts['高风险']),
        'low_risk_count': int(risk_counts['低风险']),
        'thresholds': thresholds,
    }
