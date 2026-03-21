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
        build_prediction_recommendations,
        build_single_prediction_result,
        build_student_management_view,
        get_feature_columns,
    )


class DummyScaler:
    def transform(self, values):
        return values


class DummyModel:
    classes_ = ['低风险', '中风险', '高风险']

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
        self.assertIn('data_overview = build_data_overview(df)', text)
        self.assertIn('relationship = build_feature_relationship(df, selected_feature)', text)
        self.assertIn('prediction_result = build_single_prediction_result(', text)
        self.assertIn('management_view = build_student_management_view(', text)
        self.assertIn("st.warning(\"未找到可用的模型或标准化器，请先运行训练脚本生成 Web 所需资源。\")", text)

    def test_view_model_module_exists(self):
        text = Path('src/web/view_models.py').read_text(encoding='utf-8')
        self.assertIn('def resolve_resource_paths(project_root, model_config):', text)
        self.assertIn('def load_app_resources(project_root, model_config):', text)
        self.assertIn('def build_dashboard_summary(df, grade_column=\'GRADE\'):', text)
        self.assertIn('def get_feature_columns(df, excluded_columns=None):', text)
        self.assertIn('def build_data_overview(df, grade_column=\'GRADE\'):', text)
        self.assertIn('def build_feature_relationship(df, feature_name, grade_column=\'GRADE\'):', text)
        self.assertIn('def build_student_management_view(df, model, scaler, num_high_risk, grade_column=\'GRADE\'):', text)
        self.assertIn('def build_single_prediction_result(features, model, scaler):', text)
        self.assertIn('def build_prediction_recommendations(prediction):', text)

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


if __name__ == '__main__':
    unittest.main()
