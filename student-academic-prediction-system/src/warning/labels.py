"""统一的风险标签逻辑。"""

import numpy as np
import pandas as pd

from configs.risk import RISK_LABELS, RISK_THRESHOLDS, RECOMMENDATIONS



def map_score_to_risk(score, low_threshold, high_threshold):
    if score < low_threshold:
        return RISK_LABELS[0]
    if score < high_threshold:
        return RISK_LABELS[1]
    return RISK_LABELS[2]



def build_fixed_thresholds():
    return {
        'low': float(RISK_THRESHOLDS['high_risk']),
        'high': float(RISK_THRESHOLDS['medium_risk']),
    }



def build_quantile_thresholds(y, low_quantile=0.3, high_quantile=0.7):
    values = pd.Series(y).astype(float)
    return {
        'low': float(values.quantile(low_quantile)),
        'high': float(values.quantile(high_quantile)),
    }



def classify_scores(scores, thresholds):
    series = pd.Series(scores)
    return series.apply(
        lambda score: map_score_to_risk(score, thresholds['low'], thresholds['high'])
    )



def summarize_risk_levels(scores, thresholds):
    labels = classify_scores(scores, thresholds)
    counts = labels.value_counts().reindex(RISK_LABELS, fill_value=0)
    return labels, counts



def get_recommendations(risk_label):
    return RECOMMENDATIONS.get(risk_label, [])
