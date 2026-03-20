import unittest
from pathlib import Path


class WebViewModelTests(unittest.TestCase):
    def test_web_app_imports_view_models(self):
        text = Path('app/web_app.py').read_text(encoding='utf-8')
        self.assertIn('from src.web.view_models import', text)
        self.assertIn('load_app_resources(PROJECT_ROOT, MODEL_CONFIG)', text)
        self.assertIn('dashboard_summary = build_dashboard_summary(df)', text)

    def test_view_model_module_exists(self):
        text = Path('src/web/view_models.py').read_text(encoding='utf-8')
        self.assertIn('def resolve_resource_paths(project_root, model_config):', text)
        self.assertIn('def load_app_resources(project_root, model_config):', text)
        self.assertIn('def build_dashboard_summary(df, grade_column=\'GRADE\'):', text)


if __name__ == '__main__':
    unittest.main()
