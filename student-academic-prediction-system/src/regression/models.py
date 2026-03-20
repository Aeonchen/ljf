"""回归模型定义。"""

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor



def get_default_regression_models(random_state=42):
    return {
        '线性回归': LinearRegression(),
        '岭回归': Ridge(alpha=1.0, random_state=random_state),
        'Lasso回归': Lasso(alpha=0.1, random_state=random_state),
        '决策树': DecisionTreeRegressor(max_depth=5, random_state=random_state),
        '随机森林': RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        '梯度提升': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            random_state=random_state,
        ),
        '支持向量机': SVR(kernel='rbf', C=1.0),
        'K近邻': KNeighborsRegressor(n_neighbors=5),
    }
