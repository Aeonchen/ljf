"""学业预警主流程。"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from configs.training import DATA_PATH
from src.shared.artifacts import build_manifest, save_manifest
from src.warning.features import plot_feature_importance, select_k_best_features
from src.warning.labels import build_quantile_thresholds, classify_scores
from src.warning.reporting import (
    build_comparison_rows,
    generate_warning_report,
    plot_confusion_matrix,
    plot_high_risk_analysis,
    plot_model_comparison,
    plot_risk_distribution,
    print_best_model_report,
    save_warning_artifacts,
)
from src.warning.trainer import train_warning_models


@dataclass
class WarningContext:
    X: pd.DataFrame
    y: pd.Series
    y_class: pd.Series
    thresholds: dict
    selected_features: list
    results: dict
    best_model: dict
    scaler: object



def ensure_dirs():
    for path in [
        Path('reports/warning_optimized'),
        Path('models/warning_optimized'),
    ]:
        path.mkdir(parents=True, exist_ok=True)



def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    target_col = 'GRADE'
    exclude_cols = ['STUDENT ID', 'COURSE ID', target_col]
    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return df, X, y



def identify_high_risk_students(best_model, scaler, X_all, student_ids=None, top_n=15):
    model = best_model['model']
    X_scaled = scaler.transform(X_all)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    high_risk_mask = predictions == '高风险'
    class_names = list(model.classes_)
    high_risk_idx = class_names.index('高风险') if '高风险' in class_names else 0

    indices = [idx for idx, is_high in enumerate(high_risk_mask) if is_high]
    if not indices:
        return []

    indices = sorted(indices, key=lambda i: probabilities[i, high_risk_idx], reverse=True)[:top_n]

    def prob_at(i, label):
        return probabilities[i, class_names.index(label)] if label in class_names else 0.0

    rows = []
    for rank, idx in enumerate(indices, start=1):
        rows.append({
            '排名': rank,
            '学生ID': student_ids[idx] if student_ids is not None and idx < len(student_ids) else f'STUDENT{idx + 1}',
            '风险等级': predictions[idx],
            '高风险概率': prob_at(idx, '高风险'),
            '中风险概率': prob_at(idx, '中风险'),
            '低风险概率': prob_at(idx, '低风险'),
            '特征均值': float(X_all.iloc[idx].mean()),
            '特征标准差': float(X_all.iloc[idx].std()),
        })
    return rows



def run_warning_pipeline():
    ensure_dirs()

    df, X, y = load_data()
    thresholds = build_quantile_thresholds(y)
    y_class = classify_scores(y, thresholds)

    plot_risk_distribution(y_class)

    X_selected, selected_features, feature_scores = select_k_best_features(X, y_class, k=10)
    plot_feature_importance(selected_features, feature_scores)

    training_state = train_warning_models(X_selected, y_class)
    results = training_state['results']
    best_model = training_state['best_model']
    scaler = training_state['scaler']

    comparison_rows = build_comparison_rows(results)
    plot_model_comparison(comparison_rows)

    best_result = results[best_model['name']]
    plot_confusion_matrix(best_result['y_test_true'], best_result['y_test_pred'])
    print_best_model_report(best_model['name'], best_result)

    student_ids = df['STUDENT ID'].values if 'STUDENT ID' in df.columns else None
    high_risk_students = identify_high_risk_students(best_model, scaler, X_selected, student_ids=student_ids, top_n=15)
    plot_high_risk_analysis(high_risk_students, X_selected)

    save_warning_artifacts(best_model, scaler)

    system_ctx = WarningContext(
        X=X,
        y=y,
        y_class=y_class,
        thresholds=thresholds,
        selected_features=selected_features,
        results=results,
        best_model=best_model,
        scaler=scaler,
    )
    report_path = generate_warning_report(system_ctx, high_risk_students)

    manifest = build_manifest(
        task='warning',
        model_path='models/warning_optimized/best_model.pkl',
        report_path=report_path,
        scaler_path='models/warning_optimized/scaler.pkl',
        manifest_version='1.0',
        thresholds=thresholds,
        data_path=DATA_PATH,
    )
    save_manifest(manifest, 'models/warning_optimized/manifest.json')



def main():
    run_warning_pipeline()


if __name__ == '__main__':
    main()
