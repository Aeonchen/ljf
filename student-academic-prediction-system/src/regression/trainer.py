"""回归训练编排兼容层。"""

from src.basic_models import BasicModels
from src.regression.models import get_default_regression_models


class RegressionTrainer(BasicModels):
    """在旧训练器基础上提供新的模块化入口。"""

    def initialize_models(self):
        self.models = get_default_regression_models(random_state=self.random_state)
        print(f"✅ 初始化了 {len(self.models)} 个基础模型")
        return self.models



def run_regression_training(X_train, X_test, y_train, y_test, run_visualization=True):
    trainer = RegressionTrainer(random_state=42)
    trainer.initialize_models()
    trainer.train_models(X_train, y_train, X_test, y_test)
    trainer.compare_models()
    if run_visualization:
        trainer.visualize_results(y_test, X_test)
    trainer.cross_validation(X_train, y_train, cv=5)
    try:
        trainer.hyperparameter_tuning(X_train, y_train, '随机森林')
    except Exception as exc:
        print(f"超参数调优时出错: {exc}")
    return trainer
