import unittest

from src.warning.labels import (
    build_fixed_thresholds,
    build_quantile_thresholds,
    classify_scores,
    map_score_to_risk,
)


class RiskLogicTests(unittest.TestCase):
    def test_fixed_threshold_mapping(self):
        thresholds = build_fixed_thresholds()
        self.assertEqual(map_score_to_risk(1.5, thresholds['low'], thresholds['high']), '高风险')
        self.assertEqual(map_score_to_risk(3.0, thresholds['low'], thresholds['high']), '中风险')
        self.assertEqual(map_score_to_risk(5.0, thresholds['low'], thresholds['high']), '低风险')

    def test_quantile_thresholds_are_ordered(self):
        thresholds = build_quantile_thresholds([0, 1, 2, 3, 4, 5, 6, 7])
        self.assertLessEqual(thresholds['low'], thresholds['high'])

    def test_classify_scores_uses_shared_logic(self):
        thresholds = {'low': 2.0, 'high': 5.0}
        labels = classify_scores([1.0, 3.0, 6.0], thresholds)
        self.assertEqual(labels.tolist(), ['高风险', '中风险', '低风险'])


if __name__ == '__main__':
    unittest.main()
