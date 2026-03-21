import tempfile
import unittest
from pathlib import Path

from src.shared.artifacts import build_manifest, load_manifest, save_manifest


class ArtifactManifestTests(unittest.TestCase):
    def test_manifest_round_trip(self):
        manifest = build_manifest(
            task='warning',
            model_path='models/warning_optimized/best_model.pkl',
            report_path='reports/warning_optimized/detailed_report.md',
            scaler_path='models/warning_optimized/scaler.pkl',
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / 'manifest.json'
            save_manifest(manifest, target)
            loaded = load_manifest(target)
        self.assertEqual(loaded['task'], 'warning')
        self.assertEqual(loaded['model_path'], 'models/warning_optimized/best_model.pkl')
        self.assertIn('generated_at', loaded)

    def test_missing_manifest_returns_none(self):
        self.assertIsNone(load_manifest('does-not-exist.json'))


if __name__ == '__main__':
    unittest.main()
