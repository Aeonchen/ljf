# src/broad_learning.py
"""
宽度学习系统（Broad Learning System）模块
针对小样本数据优化
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
import warnings

warnings.filterwarnings('ignore')


class BroadLearningSystem(BaseEstimator, RegressorMixin):
    def __init__(self, mapping_nodes=20, enhancement_nodes=20,  # 减少节点数
                 lambda_value=2.0, activation='sigmoid',  # 增加正则化强度
                 dropout_rate=0.3, random_state=42):  # 增加Dropout率
        """初始化BLS参数（针对小样本优化）"""
        self.mapping_nodes = mapping_nodes  # 映射节点数（减少以防止过拟合）
        self.enhancement_nodes = enhancement_nodes  # 增强节点数
        self.lambda_value = lambda_value  # 正则化参数（增加以防止过拟合）
        self.activation = activation
        self.dropout_rate = dropout_rate  # 丢弃率以防止过拟合
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.W_mapping = None
        self.W_enhancement = None
        self.b_mapping = None
        self.b_enhancement = None
        self.W_output = None

        np.random.seed(random_state)

    def _activation_function(self, x):
        """激活函数"""
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -10, 10)))  # 限制输入范围
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'softplus':
            return np.log(1 + np.exp(x))
        else:
            return x

    def _apply_dropout(self, x, training=True):
        """应用Dropout"""
        if not training or self.dropout_rate == 0:
            return x

        mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape)
        return x * mask / (1 - self.dropout_rate)

    def _init_weights_xavier(self, input_dim, output_dim):
        """Xavier权重初始化"""
        limit = np.sqrt(6 / (input_dim + output_dim))
        return np.random.uniform(-limit, limit, (input_dim, output_dim))

    def fit(self, X, y):
        """训练BLS模型（针对小样本优化）"""
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)

        input_dim = X_scaled.shape[1]

        # 初始化权重（使用Xavier初始化）
        self.W_mapping = self._init_weights_xavier(input_dim, self.mapping_nodes)
        self.b_mapping = np.zeros((1, self.mapping_nodes))

        self.W_enhancement = self._init_weights_xavier(self.mapping_nodes, self.enhancement_nodes)
        self.b_enhancement = np.zeros((1, self.enhancement_nodes))

        # 计算映射节点输出
        Z = self._activation_function(X_scaled @ self.W_mapping + self.b_mapping)
        Z = self._apply_dropout(Z, training=True)

        # 计算增强节点输出
        H = self._activation_function(Z @ self.W_enhancement + self.b_enhancement)
        H = self._apply_dropout(H, training=True)

        # 组合特征
        A = np.hstack([X_scaled, Z, H])

        # 使用岭回归求解输出权重（增加正则化）
        ridge = Ridge(alpha=self.lambda_value, random_state=self.random_state)
        ridge.fit(A, y)
        self.W_output = ridge.coef_.reshape(-1, 1)

        return self

    def predict(self, X):
        """预测"""
        X_scaled = self.scaler.transform(X)

        # 计算映射节点输出（无dropout）
        Z = self._activation_function(X_scaled @ self.W_mapping + self.b_mapping)

        # 计算增强节点输出
        H = self._activation_function(Z @ self.W_enhancement + self.b_enhancement)

        # 组合特征
        A = np.hstack([X_scaled, Z, H])

        # 预测
        predictions = A @ self.W_output

        return predictions.flatten()

    def get_feature_importance(self):
        """获取特征重要性"""
        if self.W_output is None:
            return None

        importance = np.abs(self.W_output).flatten()

        # 分类重要性
        total_original = self.W_output.shape[0] - self.mapping_nodes - self.enhancement_nodes
        if total_original > 0:
            original_importance = importance[:total_original].sum()
            mapping_importance = importance[total_original:total_original + self.mapping_nodes].sum()
            enhancement_importance = importance[total_original + self.mapping_nodes:].sum()
        else:
            original_importance = 0
            mapping_importance = 0
            enhancement_importance = importance.sum()

        return {
            'original_features': original_importance,
            'mapping_features': mapping_importance,
            'enhancement_features': enhancement_importance,
            'total': original_importance + mapping_importance + enhancement_importance
        }


class EnsembleBLS:
    """集成宽度学习系统"""

    def __init__(self, n_models=5, configs=None, random_state=42):
        """初始化多个BLS模型"""
        if configs is None:
            # 创建不同的配置以适应不同的数据模式
            configs = []
            for i in range(n_models):
                config = {
                    'mapping_nodes': np.random.randint(30, 80),
                    'enhancement_nodes': np.random.randint(30, 80),
                    'lambda_value': np.random.uniform(0.01, 0.5),
                    'activation': np.random.choice(['sigmoid', 'tanh', 'relu']),
                    'dropout_rate': np.random.uniform(0, 0.3),
                    'random_state': random_state + i
                }
                configs.append(config)

        self.models = []
        for config in configs:
            model = BroadLearningSystem(**config)
            self.models.append(model)

        self.weights = None
        self.random_state = random_state
        np.random.seed(random_state)

    def fit(self, X, y):
        """训练多个BLS模型"""
        predictions = []

        print(f"🏋️  训练 {len(self.models)} 个BLS模型...")
        for i, model in enumerate(self.models):
            print(f"  模型 {i + 1}/{len(self.models)}...")
            model.fit(X, y)
            pred = model.predict(X)
            predictions.append(pred.reshape(-1, 1))

        # 组合预测
        predictions_matrix = np.hstack(predictions)

        # 使用岭回归计算集成权重
        ridge = Ridge(alpha=0.1, random_state=self.random_state)
        ridge.fit(predictions_matrix, y)
        self.weights = ridge.coef_
        self.bias = ridge.intercept_

        print("✅ 集成BLS训练完成")
        return self

    def predict(self, X):
        """集成预测"""
        predictions = []

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred.reshape(-1, 1))

        # 组合预测
        predictions_matrix = np.hstack(predictions)

        # 加权预测
        ensemble_pred = predictions_matrix @ self.weights + self.bias

        return ensemble_pred.flatten()

    def predict_individual(self, X):
        """获取每个模型的预测"""
        predictions = {}

        for i, model in enumerate(self.models):
            predictions[f'BLS_Model_{i + 1}'] = model.predict(X)

        return predictions

    def evaluate_individual_models(self, X, y):
        """评估每个模型"""
        from sklearn.metrics import r2_score, mean_squared_error

        results = {}

        for i, model in enumerate(self.models):
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)

            results[f'BLS_Model_{i + 1}'] = {
                'R2': r2,
                'RMSE': np.sqrt(mse),
                'MSE': mse,
                'config': {
                    'mapping_nodes': model.mapping_nodes,
                    'enhancement_nodes': model.enhancement_nodes,
                    'lambda_value': model.lambda_value,
                    'activation': model.activation,
                    'dropout_rate': model.dropout_rate
                }
            }

        # 评估集成模型
        y_pred_ensemble = self.predict(X)
        r2_ensemble = r2_score(y, y_pred_ensemble)
        mse_ensemble = mean_squared_error(y, y_pred_ensemble)

        results['BLS_Ensemble'] = {
            'R2': r2_ensemble,
            'RMSE': np.sqrt(mse_ensemble),
            'MSE': mse_ensemble
        }

        return results