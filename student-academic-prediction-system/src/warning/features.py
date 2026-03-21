import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

from src.shared.plotting import save_figure, safe_close


def select_k_best_features(X, y_class, k=10):
    selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y_class)
    selected_features = X.columns[selector.get_support()].tolist()
    feature_scores = selector.scores_[selector.get_support()]
    return pd.DataFrame(X_selected, columns=selected_features), selected_features, feature_scores


def plot_feature_importance(features, scores, output_path='reports/warning_optimized/feature_importance.png'):
    if len(features) == 0:
        return None

    sorted_idx = np.argsort(scores)[::-1]
    sorted_features = [features[i] for i in sorted_idx]
    sorted_scores = scores[sorted_idx]

    fig = plt.figure(figsize=(12, 8))
    n_show = min(15, len(sorted_features))
    top_features = sorted_features[:n_show]
    top_scores = sorted_scores[:n_show]

    plt.barh(range(n_show), top_scores[::-1], color='#3498db')
    plt.yticks(range(n_show), top_features[::-1])
    plt.xlabel('特征重要性分数')
    plt.title(f'Top {n_show} 重要特征')
    plt.grid(True, alpha=0.3)

    for index, score in enumerate(top_scores[::-1]):
        plt.text(score, index, f' {score:.2f}', va='center')

    plt.tight_layout()
    save_figure(fig, output_path)
    safe_close(fig)

    return {
        'top_feature': sorted_features[0],
        'top_score': float(sorted_scores[0]),
        'selected_features': features,
    }