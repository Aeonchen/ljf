import unittest
from pathlib import Path


class WebManifestUsageTests(unittest.TestCase):
    def test_web_app_prefers_manifest_paths(self):
        text = Path('src/web/view_models.py').read_text(encoding='utf-8')
        self.assertIn('from src.shared.artifacts import load_manifest', text)
        self.assertIn("manifest_path = project_root / model_config['manifest_path']", text)
        self.assertIn("resources = {'manifest': resolved_paths['manifest'], '_status': []}", text)


if __name__ == '__main__':
    unittest.main()
