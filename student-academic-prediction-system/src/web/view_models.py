"""Streamlit 页面可复用的资源加载与视图模型。"""

import joblib
import numpy as np
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


def get_feature_columns(df, excluded_columns=None):
    excluded_columns = excluded_columns or ['GRADE', 'STUDENT ID', 'COURSE ID']
    return [col for col in df.columns if col not in excluded_columns]


def build_data_overview(df, grade_column='GRADE'):
    return {
        'row_count': int(df.shape[0]),
        'column_count': int(df.shape[1]),
        'grade_min': float(df[grade_column].min()),
        'grade_max': float(df[grade_column].max()),
        'missing_count': int(df.isnull().sum().sum()),
    }


def build_feature_relationship(df, feature_name, grade_column='GRADE'):
    feature_series = df[feature_name]
    grade_series = df[grade_column]
    relationship = {
        'feature_name': feature_name,
        'correlation': None,
        'trendline_x': None,
        'trendline_y': None,
    }

    if len(feature_series.unique()) > 1:
        z = np.polyfit(feature_series, grade_series, 1)
        p = np.poly1d(z)
        sorted_x = sorted(feature_series)
        relationship['correlation'] = float(feature_series.corr(grade_series))
        relationship['trendline_x'] = sorted_x
        relationship['trendline_y'] = p(sorted_x)

    return relationship


def build_student_management_view(df, model, scaler, num_high_risk, grade_column='GRADE'):
    feature_columns = get_feature_columns(df, excluded_columns=[grade_column, 'STUDENT ID', 'COURSE ID'])
    X = df[feature_columns[:10]]
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    results_df = df.copy()
    results_df['预测风险等级'] = predictions

    if '高风险' in model.classes_:
        high_risk_idx = list(model.classes_).index('高风险')
        results_df['高风险概率'] = probabilities[:, high_risk_idx]
    else:
        results_df['高风险概率'] = probabilities[:, 0]

    high_risk_df = results_df[results_df['预测风险等级'] == '高风险'].sort_values(
        '高风险概率', ascending=False
    ).head(num_high_risk)

    return {
        'feature_columns': feature_columns,
        'results_df': results_df,
        'high_risk_df': high_risk_df,
    }
