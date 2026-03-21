import unittest
import importlib.util
from pathlib import Path

HAS_NUMPY = importlib.util.find_spec('numpy') is not None
HAS_PANDAS = importlib.util.find_spec('pandas') is not None
HAS_VIEW_MODEL_DEPS = HAS_NUMPY and HAS_PANDAS

if HAS_VIEW_MODEL_DEPS:
    import numpy as np
    import pandas as pd

    from src.web.view_models import (
        build_data_overview,
        build_feature_relationship,
        build_feature_importance_frame,
        build_prediction_display,
        build_prediction_recommendations,
        build_prediction_probability_rows,
        build_risk_distribution,
        build_single_prediction_result,
        build_student_management_view,
        filter_student_results,
        get_feature_columns,
        get_single_prediction_feature_fields,
    )


class DummyScaler:
    def transform(self, values):
        return values


class DummyModel:
    classes_ = ['低风险', '中风险', '高风险']
    feature_importances_ = [0.1, 0.7, 0.2]

    def predict(self, values):
        return ['高风险' if row[0] >= 0.8 else '低风险' for row in values]

    def predict_proba(self, values):
        probabilities = []
        for row in values:
            high_risk_prob = float(row[0])
            probabilities.append([1 - high_risk_prob, 0.0, high_risk_prob])
        return np.array(probabilities)


class WebViewModelTests(unittest.TestCase):
    def test_web_app_imports_view_models(self):
        text = Path('app/web_app.py').read_text(encoding='utf-8')
        self.assertIn('from src.web.view_models import', text)
        self.assertIn('load_app_resources(PROJECT_ROOT, MODEL_CONFIG)', text)
        self.assertIn('dashboard_summary = build_dashboard_summary(df)', text)
        self.assertIn('risk_distribution = build_risk_distribution(risk_labels)', text)
        self.assertIn('top_features = build_feature_importance_frame(model)', text)
        self.assertIn('data_overview = build_data_overview(df)', text)
        self.assertIn('relationship = build_feature_relationship(df, selected_feature)', text)
        self.assertIn('for column, field_group in zip(st.columns(4), get_single_prediction_feature_fields()):', text)
        self.assertIn('prediction_result = build_single_prediction_result(', text)
        self.assertIn('prediction_display = build_prediction_display(prediction)', text)
        self.assertIn('probability_rows = build_prediction_probability_rows(', text)
        self.assertIn('management_view = build_student_management_view(', text)
        self.assertIn('student_data = filter_student_results(results_df, student_id=student_id)', text)
        self.assertIn("st.warning(\"未找到可用的模型或标准化器，请先运行训练脚本生成 Web 所需资源。\")", text)

    def test_view_model_module_exists(self):
        text = Path('src/web/view_models.py').read_text(encoding='utf-8')
        self.assertIn('def resolve_resource_paths(project_root, model_config):', text)
        self.assertIn('def load_app_resources(project_root, model_config):', text)
        self.assertIn('def build_dashboard_summary(df, grade_column=\'GRADE\'):', text)
        self.assertIn('def build_risk_distribution(risk_labels):', text)
        self.assertIn('def build_feature_importance_frame(model, top_n=10):', text)
        self.assertIn('def get_feature_columns(df, excluded_columns=None):', text)
        self.assertIn('def build_data_overview(df, grade_column=\'GRADE\'):', text)
        self.assertIn('def build_feature_relationship(df, feature_name, grade_column=\'GRADE\'):', text)
        self.assertIn('def build_student_management_view(df, model, scaler, num_high_risk, grade_column=\'GRADE\'):', text)
        self.assertIn('def get_single_prediction_feature_fields():', text)
        self.assertIn('def build_single_prediction_result(features, model, scaler):', text)
        self.assertIn('def build_prediction_probability_rows(classes, probabilities):', text)
        self.assertIn('def build_prediction_display(prediction):', text)
        self.assertIn('def build_prediction_recommendations(prediction):', text)
        self.assertIn('def filter_student_results(results_df, student_id=None, min_grade=None, max_grade=None, grade_column=\'GRADE\'):', text)

    @unittest.skipUnless(HAS_VIEW_MODEL_DEPS, 'numpy/pandas not installed in test environment')
    def test_overview_and_feature_helpers_return_expected_values(self):
        df = pd.DataFrame({
            'STUDENT ID': ['S1', 'S2', 'S3'],
            'COURSE ID': ['C1', 'C1', 'C1'],
            'GRADE': [1.0, 2.0, 3.0],
            'feature_a': [0.2, 0.4, 0.6],
            'feature_b': [1.0, 1.0, 1.0],
        })

        self.assertEqual(get_feature_columns(df), ['feature_a', 'feature_b'])

        overview = build_data_overview(df)
        self.assertEqual(overview['row_count'], 3)
        self.assertEqual(overview['column_count'], 5)
        self.assertEqual(overview['grade_min'], 1.0)
        self.assertEqual(overview['grade_max'], 3.0)

        relationship = build_feature_relationship(df, 'feature_a')
        self.assertAlmostEqual(relationship['correlation'], 1.0)
        self.assertIsNotNone(relationship['trendline_x'])
        self.assertIsNotNone(relationship['trendline_y'])

        flat_relationship = build_feature_relationship(df, 'feature_b')
        self.assertIsNone(flat_relationship['correlation'])
        self.assertIsNone(flat_relationship['trendline_x'])

    @unittest.skipUnless(HAS_VIEW_MODEL_DEPS, 'numpy/pandas not installed in test environment')
    def test_student_management_view_builds_ranked_high_risk_rows(self):
        df = pd.DataFrame({
            'STUDENT ID': ['S1', 'S2', 'S3'],
            'COURSE ID': ['C1', 'C1', 'C1'],
            'GRADE': [1.0, 2.0, 3.0],
            'feature_1': [0.9, 0.2, 0.8],
            'feature_2': [0.0, 0.0, 0.0],
            'feature_3': [0.0, 0.0, 0.0],
            'feature_4': [0.0, 0.0, 0.0],
            'feature_5': [0.0, 0.0, 0.0],
            'feature_6': [0.0, 0.0, 0.0],
            'feature_7': [0.0, 0.0, 0.0],
            'feature_8': [0.0, 0.0, 0.0],
            'feature_9': [0.0, 0.0, 0.0],
            'feature_10': [0.0, 0.0, 0.0],
        })

        view = build_student_management_view(df, DummyModel(), DummyScaler(), 2)

        self.assertEqual(list(view['results_df']['预测风险等级']), ['高风险', '低风险', '高风险'])
        self.assertEqual(list(view['high_risk_df']['STUDENT ID']), ['S1', 'S3'])
        self.assertGreaterEqual(view['high_risk_df'].iloc[0]['高风险概率'], view['high_risk_df'].iloc[1]['高风险概率'])

    @unittest.skipUnless(HAS_VIEW_MODEL_DEPS, 'numpy/pandas not installed in test environment')
    def test_single_prediction_result_returns_probabilities_and_recommendations(self):
        features = {
            'feature_29': 0.9,
            'feature_14': 0.0,
            'feature_28': 0.0,
            'feature_12': 0.0,
            'feature_11': 0.0,
            'feature_9': 0.0,
            'feature_5': 0.0,
            'feature_3': 0.0,
        }

        result = build_single_prediction_result(features, DummyModel(), DummyScaler())

        self.assertEqual(result['prediction'], '高风险')
        self.assertEqual(result['classes'], ['低风险', '中风险', '高风险'])
        self.assertAlmostEqual(float(result['probabilities'][2]), 0.9)
        self.assertIn('立即安排一对一辅导', result['recommendations'])

    @unittest.skipUnless(HAS_VIEW_MODEL_DEPS, 'numpy/pandas not installed in test environment')
    def test_prediction_recommendations_cover_low_and_medium_risk(self):
        self.assertIn('每月评估学习进步', build_prediction_recommendations('中风险'))
        self.assertIn('保持当前学习节奏', build_prediction_recommendations('低风险'))

    @unittest.skipUnless(HAS_VIEW_MODEL_DEPS, 'numpy/pandas not installed in test environment')
    def test_visualization_helpers_prepare_risk_and_probability_data(self):
        risk_distribution = build_risk_distribution(pd.Series(['高风险', '低风险', '高风险']))
        self.assertEqual(risk_distribution['labels'], ['高风险', '中风险', '低风险'])
        self.assertEqual(list(risk_distribution['counts']), [2, 0, 1])

        feature_importance = build_feature_importance_frame(DummyModel(), top_n=2)
        self.assertEqual(list(feature_importance['特征']), ['特征2', '特征3'])

        probability_rows = build_prediction_probability_rows(
            ['低风险', '中风险', '高风险'],
            np.array([0.2, 0.3, 0.5]),
        )
        self.assertEqual(list(probability_rows['label']), ['低风险', '中风险', '高风险'])
        self.assertEqual(list(probability_rows['probability']), [20.0, 30.0, 50.0])

    @unittest.skipUnless(HAS_VIEW_MODEL_DEPS, 'numpy/pandas not installed in test environment')
    def test_prediction_form_and_filter_helpers_prepare_expected_values(self):
        field_groups = get_single_prediction_feature_fields()
        self.assertEqual(len(field_groups), 4)
        self.assertEqual(field_groups[0][0], ('feature_29', '特征29'))

        display = build_prediction_display('中风险')
        self.assertEqual(display['css_class'], 'risk-medium')
        self.assertEqual(display['icon'], '🟡')

        results_df = pd.DataFrame({
            'STUDENT ID': ['S1', 'S2', 'S3'],
            'GRADE': [1.0, 2.5, 3.5],
        })
        student_rows = filter_student_results(results_df, student_id='S2')
        self.assertEqual(list(student_rows['STUDENT ID']), ['S2'])

        grade_rows = filter_student_results(results_df, min_grade=2.0, max_grade=3.0)
        self.assertEqual(list(grade_rows['STUDENT ID']), ['S2'])


if __name__ == '__main__':
    unittest.main()
