# src/feature_engineering.py
"""
特征工程模块
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineering:
    """特征工程类 - 针对学生学业数据优化"""

    def __init__(self, config=None):
        if config is None:
            from config import FEATURE_ENGINEERING
            self.config = FEATURE_ENGINEERING
        else:
            self.config = config

        self.poly_transformer = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.pca = None
        self.kmeans = None
        self.selected_features = []
        self.feature_importance = None

    def analyze_features(self, X, y):
        """特征分析"""
        print("🔍 特征分析...")

        results = {}

        # 1. 方差分析
        variances = X.var()
        low_variance_features = variances[variances < 0.01].index.tolist()

        results['variance_analysis'] = {
            'total_features': len(X.columns),
            'low_variance_features': low_variance_features,
            'low_variance_count': len(low_variance_features)
        }

        print(f"  发现 {len(low_variance_features)} 个低方差特征")

        # 2. 相关性分析
        correlations = {}
        for col in X.columns:
            corr = np.corrcoef(X[col], y)[0, 1]
            correlations[col] = corr

        # 找出最相关的特征
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        top_positive = [feat for feat, corr in sorted_corr[:5] if corr > 0]
        top_negative = [feat for feat, corr in sorted_corr[:5] if corr < 0]

        results['correlation_analysis'] = {
            'top_positive': top_positive,
            'top_negative': top_negative,
            'correlations': correlations
        }

        print(f"  与目标最正相关的特征: {top_positive[:3]}")
        print(f"  与目标最负相关的特征: {top_negative[:3]}")

        # 3. 互信息分析
        try:
            mi_scores = mutual_info_regression(X, y, random_state=42)
            mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
            top_mi_features = mi_series.head(5).index.tolist()

            results['mutual_information'] = {
                'top_features': top_mi_features,
                'scores': mi_series.to_dict()
            }

            print(f"  互信息最高的特征: {top_mi_features}")
        except:
            print("⚠️  无法计算互信息")

        return results

    def create_polynomial_features(self, X, degree=2):
        """创建多项式特征（针对小样本优化）"""
        print(f"🔧 创建 {degree} 阶多项式特征...")

        # 对于小样本数据，只创建交互特征，避免维度爆炸
        poly = PolynomialFeatures(
            degree=degree,
            include_bias=False,
            interaction_only=True  # 只创建交互特征
        )

        X_poly = poly.fit_transform(X)
        poly_feature_names = poly.get_feature_names_out(X.columns)

        # 转换为DataFrame
        X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)

        # 过滤低方差特征
        selector = VarianceThreshold(threshold=0.01)
        X_selected = selector.fit_transform(X_poly_df)
        selected_mask = selector.get_support()
        selected_features = X_poly_df.columns[selected_mask].tolist()

        X_final = pd.DataFrame(X_selected, columns=selected_features)

        print(f"✅ 多项式特征: {X.shape[1]} → {X_final.shape[1]} 个特征")
        self.poly_transformer = poly

        return X_final

    def create_interaction_features(self, X, y):
        """创建有意义的交互特征"""
        print("🔗 创建交互特征...")

        # 找出最相关的特征
        correlations = {}
        for col in X.columns:
            corr = np.corrcoef(X[col], y)[0, 1]
            correlations[col] = abs(corr)

        # 选择最相关的特征进行交互
        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:8]
        top_feature_names = [feat for feat, _ in top_features]

        # 创建交互特征
        interaction_features = pd.DataFrame()

        for i in range(len(top_feature_names)):
            for j in range(i + 1, len(top_feature_names)):
                feat1 = top_feature_names[i]
                feat2 = top_feature_names[j]
                interaction_name = f"{feat1}_x_{feat2}"
                interaction_features[interaction_name] = X[feat1] * X[feat2]

                # 创建比率特征
                if (X[feat2] != 0).all():  # 避免除以0
                    ratio_name = f"{feat1}_div_{feat2}"
                    interaction_features[ratio_name] = X[feat1] / (X[feat2] + 1e-10)  # 添加小值避免除零

        print(f"✅ 创建了 {interaction_features.shape[1]} 个交互特征")
        return interaction_features

    def create_statistical_features(self, X):
        """创建统计特征"""
        print("📊 创建统计特征...")

        # 分组特征创建统计量
        statistical_features = pd.DataFrame()

        # 1. 特征分组（根据特征含义）
        # 假设特征1-10是基本信息，11-20是家庭背景，21-30是学习习惯
        basic_features = [col for col in X.columns if col in [str(i) for i in range(1, 11)]]
        family_features = [col for col in X.columns if col in [str(i) for i in range(11, 21)]]
        study_features = [col for col in X.columns if col in [str(i) for i in range(21, 31)]]

        # 2. 计算每组的统计特征
        if basic_features:
            statistical_features['basic_mean'] = X[basic_features].mean(axis=1)
            statistical_features['basic_std'] = X[basic_features].std(axis=1)

        if family_features:
            statistical_features['family_mean'] = X[family_features].mean(axis=1)
            statistical_features['family_range'] = X[family_features].max(axis=1) - X[family_features].min(axis=1)

        if study_features:
            statistical_features['study_mean'] = X[study_features].mean(axis=1)
            statistical_features['study_sum'] = X[study_features].sum(axis=1)

        # 3. 整体统计特征
        statistical_features['all_mean'] = X.mean(axis=1)
        statistical_features['all_std'] = X.std(axis=1)
        statistical_features['all_skew'] = X.skew(axis=1)

        print(f"✅ 创建了 {statistical_features.shape[1]} 个统计特征")
        return statistical_features

    def create_cluster_features(self, X, n_clusters=3):
        """创建聚类特征"""
        print(f"🎯 创建 {n_clusters} 个聚类特征...")

        # 使用PCA降维后再聚类
        pca = PCA(n_components=min(5, X.shape[1]))
        X_pca = pca.fit_transform(X)

        # K-means聚类
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_pca)

        # 创建聚类特征
        cluster_features = pd.get_dummies(cluster_labels, prefix='cluster')

        # 添加聚类中心距离特征
        cluster_centers = self.kmeans.cluster_centers_
        for i in range(n_clusters):
            distances = np.linalg.norm(X_pca - cluster_centers[i], axis=1)
            cluster_features[f'dist_to_cluster_{i}'] = distances

        print(f"✅ 创建了 {cluster_features.shape[1]} 个聚类特征")
        return cluster_features

    def select_features_mutual_info(self, X, y, k=15):
        """使用互信息进行特征选择"""
        print("🎯 使用互信息进行特征选择...")

        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

        # 选择top k特征
        selected_features = mi_series.head(min(k, len(mi_series))).index.tolist()

        self.selected_features = selected_features
        self.feature_importance = mi_series

        print(f"✅ 选择了 {len(selected_features)} 个特征")
        print(f"  最重要的特征: {selected_features[:5]}")

        return X[selected_features], selected_features, mi_series

    def select_features_ensemble(self, X, y, n_features=15):
        """使用集成方法进行特征选择"""
        print("🏆 使用集成方法进行特征选择...")

        # 方法1: 随机森林特征重要性
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = pd.Series(rf.feature_importances_, index=X.columns)

        # 方法2: 互信息
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_importance = pd.Series(mi_scores, index=X.columns)

        # 方法3: 相关性
        correlations = pd.Series({col: abs(np.corrcoef(X[col], y)[0, 1]) for col in X.columns})

        # 组合三种方法
        combined_scores = (
                rf_importance.rank() +
                mi_importance.rank() +
                correlations.rank()
        )

        # 选择top特征
        selected_features = combined_scores.sort_values(ascending=False).head(n_features).index.tolist()

        self.selected_features = selected_features
        self.feature_importance = combined_scores

        print(f"✅ 集成特征选择完成: 选择 {len(selected_features)} 个特征")

        return X[selected_features], selected_features, combined_scores

    def apply_pca_optimized(self, X, variance_threshold=0.95):
        """优化PCA：保留指定方差比例"""
        print(f"📉 应用优化PCA (保留{variance_threshold * 100:.0f}%方差)...")

        # 标准化数据
        X_scaled = self.scaler.fit_transform(X)

        # 逐步增加主成分直到达到方差阈值
        max_components = min(X.shape[1], 20)
        best_n_components = 1

        for n in range(1, max_components + 1):
            pca = PCA(n_components=n)
            pca.fit(X_scaled)
            explained_variance = pca.explained_variance_ratio_.sum()

            if explained_variance >= variance_threshold:
                best_n_components = n
                break

        # 应用PCA
        self.pca = PCA(n_components=best_n_components)
        X_pca = self.pca.fit_transform(X_scaled)

        print(f"✅ PCA完成: {X.shape[1]} → {best_n_components} 个主成分")
        print(f"📊 解释方差比: {self.pca.explained_variance_ratio_.sum():.3f}")

        return X_pca, self.pca.explained_variance_ratio_

    # 在 src/feature_engineering.py 中修改 run_full_pipeline 函数：

    def run_full_pipeline(self, X_train, X_test, y_train, visualize=True):
        """运行完整的特征工程管道"""
        print("=" * 60)
        print("🚀 特征工程管道")
        print("=" * 60)

        results = {}

        # 1. 特征分析
        print("\n📝 步骤1: 特征分析")
        analysis_results = self.analyze_features(X_train, y_train)
        results['analysis'] = analysis_results

        # 2. 创建多项式特征
        print("\n🔧 步骤2: 创建多项式特征")
        if self.config.get('polynomial_degree', 1) > 1:
            X_train_poly = self.create_polynomial_features(X_train, self.config.get('polynomial_degree', 2))
            X_test_poly = self.poly_transformer.transform(X_test)

            # 保持相同的列
            X_test_poly = pd.DataFrame(X_test_poly,
                                       columns=self.poly_transformer.get_feature_names_out(X_train.columns))
            X_test_poly = X_test_poly[X_train_poly.columns]  # 确保列一致
        else:
            X_train_poly = X_train.copy()
            X_test_poly = X_test.copy()

        # 3. 创建交互特征
        print("\n🔗 步骤3: 创建交互特征")
        if self.config.get('interaction_features', True):
            X_train_interaction = self.create_interaction_features(X_train, y_train)

            # 为测试集创建相同的交互特征
            X_test_interaction_df = pd.DataFrame()
            for col in X_train_interaction.columns:
                if '_x_' in col:
                    feat1, feat2 = col.split('_x_')
                    if feat1 in X_test.columns and feat2 in X_test.columns:
                        X_test_interaction_df[col] = X_test[feat1] * X_test[feat2]
                elif '_div_' in col:
                    feat1, feat2 = col.split('_div_')
                    if feat1 in X_test.columns and feat2 in X_test.columns:
                        X_test_interaction_df[col] = X_test[feat1] / (X_test[feat2] + 1e-10)

            X_train_enhanced = pd.concat([X_train_poly, X_train_interaction], axis=1)
            X_test_enhanced = pd.concat([X_test_poly, X_test_interaction_df], axis=1)
        else:
            X_train_enhanced = X_train_poly
            X_test_enhanced = X_test_poly

        # 4. 创建统计特征
        print("\n📈 步骤4: 创建统计特征")
        if self.config.get('create_statistical_features', True):
            X_train_stats = self.create_statistical_features(X_train)
            X_test_stats = self.create_statistical_features(X_test)

            X_train_enhanced = pd.concat([X_train_enhanced, X_train_stats], axis=1)
            X_test_enhanced = pd.concat([X_test_enhanced, X_test_stats], axis=1)

        # 5. 创建聚类特征
        print("\n🎯 步骤5: 创建聚类特征")
        if self.config.get('create_cluster_features', False):
            try:
                n_clusters = self.config.get('n_clusters', 3)
                X_train_clusters = self.create_cluster_features(X_train_enhanced, n_clusters)

                # 为测试集创建相同的聚类特征
                if self.kmeans is not None:
                    # 首先确保pca对象存在
                    if self.pca is not None:
                        # 使用相同的PCA和Scaler转换测试集
                        X_test_scaled_for_cluster = self.scaler.transform(X_test_enhanced)
                        X_test_pca_for_cluster = self.pca.transform(X_test_scaled_for_cluster)

                        test_cluster_labels = self.kmeans.predict(X_test_pca_for_cluster)
                        X_test_clusters = pd.get_dummies(test_cluster_labels, prefix='cluster')

                        # 添加距离特征
                        cluster_centers = self.kmeans.cluster_centers_
                        for i in range(n_clusters):
                            distances = np.linalg.norm(X_test_pca_for_cluster - cluster_centers[i], axis=1)
                            X_test_clusters[f'dist_to_cluster_{i}'] = distances

                        X_train_enhanced = pd.concat([X_train_enhanced, X_train_clusters], axis=1)
                        X_test_enhanced = pd.concat([X_test_enhanced, X_test_clusters], axis=1)
                    else:
                        print("⚠️  PCA对象不存在，跳过为测试集创建聚类特征")
                        X_train_enhanced = pd.concat([X_train_enhanced, X_train_clusters], axis=1)
                else:
                    print("⚠️  KMeans对象不存在，跳过聚类特征")
            except Exception as e:
                print(f"⚠️  创建聚类特征时出错: {e}")
                import traceback
                traceback.print_exc()

        # 6. 特征选择
        print("\n🎯 步骤6: 特征选择")
        if self.config.get('top_k_features'):
            k = self.config['top_k_features']
            X_train_selected, selected_features, importance_scores = self.select_features_ensemble(
                X_train_enhanced, y_train, k
            )
            X_test_selected = X_test_enhanced[selected_features]

            results['feature_selection'] = {
                'selected_features': selected_features,
                'importance_scores': importance_scores.to_dict(),
                'train_shape': X_train_selected.shape,
                'test_shape': X_test_selected.shape
            }
        else:
            X_train_selected = X_train_enhanced
            X_test_selected = X_test_enhanced

        # 7. PCA降维
        print("\n📊 步骤7: PCA降维")
        if self.config.get('pca_components'):
            try:
                X_train_pca, variance_ratio = self.apply_pca_optimized(X_train_selected, 0.95)

                # 确保PCA对象已创建
                if self.pca is not None:
                    # 对测试集应用相同的PCA转换
                    X_test_scaled = self.scaler.transform(X_test_selected)
                    X_test_pca = self.pca.transform(X_test_scaled)

                    results['pca'] = {
                        'explained_variance': variance_ratio.sum(),
                        'train_shape': X_train_pca.shape,
                        'test_shape': X_test_pca.shape
                    }

                    X_train_final = pd.DataFrame(X_train_pca)
                    X_test_final = pd.DataFrame(X_test_pca)
                else:
                    print("⚠️  PCA对象创建失败，跳过PCA降维")
                    X_train_final = X_train_selected
                    X_test_final = X_test_selected
            except Exception as e:
                print(f"⚠️  PCA降维时出错: {e}")
                X_train_final = X_train_selected
                X_test_final = X_test_selected
        else:
            X_train_final = X_train_selected
            X_test_final = X_test_selected

        # 8. 特征缩放
        print("\n📏 步骤8: 特征缩放")
        if self.config.get('feature_scaling', True):
            # 如果已经应用了PCA，数据已经标准化过，不再重复标准化
            if not self.config.get('pca_components') or self.pca is None:
                X_train_scaled = self.scaler.fit_transform(X_train_final)
                X_test_scaled = self.scaler.transform(X_test_final)

                X_train_final = pd.DataFrame(X_train_scaled, columns=X_train_final.columns if hasattr(X_train_final,
                                                                                                      'columns') else None)
                X_test_final = pd.DataFrame(X_test_scaled,
                                            columns=X_test_final.columns if hasattr(X_test_final, 'columns') else None)

        print("\n" + "=" * 60)
        print("✅ 特征工程完成！")
        print(f"   原始特征数: {X_train.shape[1]}")
        print(f"   最终特征数: {X_train_final.shape[1]}")
        print(f"   训练集形状: {X_train_final.shape}")
        print(f"   测试集形状: {X_test_final.shape}")

        results['summary'] = {
            'original_features': X_train.shape[1],
            'final_features': X_train_final.shape[1],
            'train_shape': X_train_final.shape,
            'test_shape': X_test_final.shape
        }

        # 可视化特征重要性
        if visualize and self.feature_importance is not None:
            self.visualize_feature_importance()

        return X_train_final, X_test_final, results

    def visualize_feature_importance(self, top_n=20):
        """可视化特征重要性"""
        import matplotlib.pyplot as plt

        if self.feature_importance is None:
            return

        # 取前N个重要特征
        top_features = self.feature_importance.head(top_n)

        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))

        bars = plt.barh(range(len(top_features)), top_features.values, color=colors)
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('特征重要性分数')
        plt.title(f'特征重要性（前{top_n}）')
        plt.grid(True, alpha=0.3)

        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, top_features.values)):
            plt.text(value, i, f' {value:.3f}', va='center')

        plt.tight_layout()
        plt.savefig('reports/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("📊 特征重要性图已保存到 reports/feature_importance.png")