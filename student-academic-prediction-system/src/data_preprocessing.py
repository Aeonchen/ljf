"""
阶段1：数据预处理模块
核心功能：加载、清洗、预处理数据
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from configs.training import DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, FEATURE_NAMES
from src.shared.io import load_csv

class DataPreprocessor:
    """数据预处理器 - 阶段1简化版"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None

    def load_data(self, filepath=None):
        """加载数据"""
        filepath = filepath or DATA_PATH

        try:
            df = load_csv(filepath)

            print(f"✅ 数据加载成功！形状: {df.shape}")
            print(f"📊 数据预览:")
            print(df.head())

            return df

        except FileNotFoundError:
            print(f"❌ 文件未找到: {filepath}")
            print("请确保数据文件位于 data/DATA (1).csv")
            raise

        except Exception as e:
            print(f"❌ 加载数据时出错: {str(e)}")
            raise

    def explore_data(self, df):
        """数据探索"""
        print("=" * 50)
        print("🔍 数据探索")
        print("=" * 50)

        # 基本信息
        print(f"数据集形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

        # 数据类型
        print("\n📝 数据类型:")
        print(df.dtypes)

        # 缺失值
        print("\n⚠️  缺失值统计:")
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            '缺失数量': missing,
            '缺失比例(%)': missing_percent
        })
        print(missing_df[missing_df['缺失数量'] > 0])

        # 描述性统计
        print("\n📈 数值列描述性统计:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols].describe())

        # 分类列统计
        print("\n🏷️  分类列统计:")
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols[:5]:  # 只显示前5个
            print(f"\n{col}:")
            print(df[col].value_counts().head())

        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'missing_values': missing_df,
            'numeric_cols': list(numeric_cols),
            'categorical_cols': list(categorical_cols)
        }

    def clean_data(self, df):
        """数据清洗"""
        df_clean = df.copy()
        print("\n🧹 开始数据清洗...")

        # 1. 处理列名空格
        df_clean.columns = df_clean.columns.str.strip()

        # 2. 查找目标列
        target_col = None
        possible_targets = [TARGET_COLUMN, 'grade', 'Grade', 'GPA', 'gpa', '成绩', '分数']
        for col in df_clean.columns:
            if any(target in str(col).lower() for target in [t.lower() for t in possible_targets]):
                target_col = col
                print(f"🎯 找到目标列: {target_col}")
                break

        if target_col is None:
            print("⚠️  警告：未找到明确的目标列，尝试推断...")
            # 假设最后一列是目标列
            target_col = df_clean.columns[-1]
            print(f"🤔 推断目标列为: {target_col}")

        # 3. 识别特征列
        feature_cols = [col for col in df_clean.columns if col != target_col]
        numeric_features = df_clean[feature_cols].select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 5:  # 如果有足够多的数值特征
            feature_cols = list(numeric_features)

        print(f"🔢 识别到 {len(feature_cols)} 个特征列")

        # 4. 处理缺失值
        missing_before = df_clean.isnull().sum().sum()

        # 对数值列用中位数填充
        for col in feature_cols + [target_col]:
            if col in df_clean.columns:
                if df_clean[col].dtype in [np.float64, np.int64]:
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                    print(f"  用中位数填充 {col}: {median_val:.2f}")
                else:
                    # 对分类列用众数填充
                    mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    print(f"  用众数填充 {col}: {mode_val}")

        missing_after = df_clean.isnull().sum().sum()
        print(f"✅ 缺失值处理完成: {missing_before} -> {missing_after}")

        # 5. 处理异常值（简单的IQR方法）
        numeric_features = df_clean[feature_cols].select_dtypes(include=[np.number]).columns
        outliers_count = 0

        for col in numeric_features:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
            outliers_count += len(outliers)

            # 缩尾处理
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

        if outliers_count > 0:
            print(f"⚠️  处理了 {outliers_count} 个异常值（缩尾处理）")

        # 6. 检查目标列类型
        if df_clean[target_col].dtype == 'object':
            print(f"🔄 目标列 {target_col} 是文本类型，尝试转换为数值...")
            try:
                df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
                # 再次填充转换失败的缺失值
                median_val = df_clean[target_col].median()
                df_clean[target_col] = df_clean[target_col].fillna(median_val)
                print(f"✅ 目标列已转换为数值类型")
            except (TypeError, ValueError) as exc:
                print(f"⚠️  无法转换为数值，保持原样: {exc}")

        # 保存清洗后的数据
        self.clean_df = df_clean
        self.target_column = target_col
        self.feature_columns = feature_cols

        print(f"✅ 数据清洗完成！")
        print(f"  目标列: {target_col}")
        print(f"  特征列数: {len(feature_cols)}")
        print(f"  数据形状: {df_clean.shape}")

        return df_clean

    def prepare_features(self, df):
        """准备特征矩阵和目标向量"""
        print("\n🔧 准备特征矩阵和目标向量...")

        # 确保使用清洗后的数据
        if not hasattr(self, 'clean_df'):
            df = self.clean_data(df)
        else:
            df = self.clean_df

        # 提取特征和目标
        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()

        # 特征名称映射
        self.feature_names = {}
        for col in self.feature_columns:
            if 'FEATURE_NAMES' in globals() and col in FEATURE_NAMES:
                self.feature_names[col] = FEATURE_NAMES[col]
            else:
                self.feature_names[col] = col

        print(f"✅ 特征矩阵 X: {X.shape}")
        print(f"✅ 目标向量 y: {y.shape}")
        print(f"✅ 特征数量: {len(self.feature_columns)}")

        # 显示目标变量分布
        print(f"\n🎯 目标变量统计:")
        print(f"  平均值: {y.mean():.2f}")
        print(f"  标准差: {y.std():.2f}")
        print(f"  最小值: {y.min():.2f}")
        print(f"  最大值: {y.max():.2f}")
        print(f"  中位数: {y.median():.2f}")

        return X, y

    def scale_features(self, X_train, X_test=None):
        """标准化特征"""
        print("\n📏 标准化特征...")

        # 训练集标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

        # 测试集标准化（如果提供）
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            print(f"✅ 标准化完成: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
            return X_train_scaled, X_test_scaled
        else:
            print(f"✅ 标准化完成: 训练集 {X_train.shape}")
            return X_train_scaled

    def split_data(self, X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
        """划分训练集和测试集"""
        print(f"\n✂️  划分数据集: 测试集比例={test_size}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"✅ 训练集: X={X_train.shape}, y={y_train.shape}")
        print(f"✅ 测试集: X={X_test.shape}, y={y_test.shape}")

        return X_train, X_test, y_train, y_test

    def run_pipeline(self, data_path=None):
        """运行完整的数据预处理流程"""
        print("=" * 60)
        print("🚀 开始数据预处理流程")
        print("=" * 60)

        # 1. 加载数据
        df = self.load_data(data_path)

        # 2. 数据探索
        self.explore_data(df)

        # 3. 数据清洗
        df_clean = self.clean_data(df)

        # 4. 准备特征和目标
        X, y = self.prepare_features(df_clean)

        # 5. 划分数据集
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # 6. 标准化特征
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        print("\n" + "=" * 60)
        print("🎉 数据预处理完成！")
        print("=" * 60)

        return X_train_scaled, X_test_scaled, y_train, y_test


def check_data_compatibility():
    """检查数据兼容性"""
    import os

    if not os.path.exists("data"):
        print("📁 创建 data 目录")
        os.makedirs("data", exist_ok=True)

    if os.path.exists(DATA_PATH):
        print(f"✅ 找到数据文件: {DATA_PATH}")

        # 尝试加载数据查看结构
        try:
            df = pd.read_csv(DATA_PATH, nrows=5)
            print(f"✅ 数据预览（前5行）:")
            print(df.head())
            print(f"\n📊 数据列名: {list(df.columns)}")
            return True
        except Exception as e:
            print(f"❌ 读取数据文件时出错: {str(e)}")
            return False
    else:
        print(f"❌ 未找到数据文件: {DATA_PATH}")
        print("请将您的数据文件命名为 'DATA (1).csv' 并放置在 data/ 目录下")
        return False


if __name__ == "__main__":
    # 检查数据
    if check_data_compatibility():
        # 运行预处理
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.run_pipeline()

        print("\n🎯 预处理结果:")
        print(f"训练集: X={X_train.shape}, y={y_train.shape}")
        print(f"测试集: X={X_test.shape}, y={y_test.shape}")
        print(f"特征数量: {X_train.shape[1]}")
    else:
        print("\n❌ 请先准备数据文件")