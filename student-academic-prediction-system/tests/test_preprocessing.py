import unittest

from src.data_preprocessing import DataPreprocessor


class PreprocessingTests(unittest.TestCase):
    def test_pipeline_shapes(self):
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.run_pipeline()
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertEqual(len(y_train) + len(y_test), 145)


if __name__ == '__main__':
    unittest.main()
