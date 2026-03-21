import unittest
from pathlib import Path


class WarningEntrypointTests(unittest.TestCase):
    def test_run_warning_uses_pipeline(self):
        text = Path('run_warning.py').read_text(encoding='utf-8')
        self.assertIn('from src.warning.pipeline import main', text)

    def test_warning_pipeline_delegates_to_legacy_main(self):
        text = Path('src/warning/pipeline.py').read_text(encoding='utf-8')
        self.assertIn('from run_warning_optimized import main as legacy_warning_main', text)
        self.assertIn('legacy_warning_main()', text)


if __name__ == '__main__':
    unittest.main()
