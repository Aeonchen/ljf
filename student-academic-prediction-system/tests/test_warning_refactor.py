import unittest
from pathlib import Path


class WarningRefactorTests(unittest.TestCase):
    def test_warning_modules_exist(self):
        for path in [
            'src/warning/features.py',
            'src/warning/trainer.py',
            'src/warning/reporting.py',
        ]:
            self.assertTrue(Path(path).exists(), path)

    def test_run_warning_optimized_uses_modular_helpers(self):
        text = Path('run_warning_optimized.py').read_text(encoding='utf-8')
        self.assertIn('from src.warning.features import', text)
        self.assertIn('from src.warning.reporting import', text)
        self.assertIn('from src.warning.trainer import train_warning_models', text)
        self.assertIn('training_state = train_warning_models', text)
        self.assertIn('report_path = generate_warning_report(self, high_risk_students)', text)


if __name__ == '__main__':
    unittest.main()
