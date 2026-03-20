"""预警模型训练与评估。"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from configs.training import RANDOM_STATE



def get_warning_models():
    return {
        '逻辑回归_优化': LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE,
        ),
        '随机森林_优化': RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        '梯度提升_优化': GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        'K近邻_优化': KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
        ),
    }



def train_warning_models(X, y_class, scaler=None):
    scaler = scaler or StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=RANDOM_STATE, stratify=y_class
    )

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    results = {}

    for name, model in get_warning_models().items():
        model.fit(X_train_scaled, y_train)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        cv_scores = cross_val_score(
            model,
            X_train_scaled,
            y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring='accuracy',
            n_jobs=-1,
        )
        results[name] = {
            'model': model,
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'precision': precision_score(y_test, y_test_pred, average='weighted'),
            'recall': recall_score(y_test, y_test_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_test_pred, average='weighted'),
            'y_test_pred': y_test_pred,
            'y_test_true': y_test,
        }

    best_model_name = max(
        results,
        key=lambda model_name: results[model_name]['test_accuracy'] * 0.6 + results[model_name]['cv_mean'] * 0.4,
    )
    best_model = {
        'name': best_model_name,
        'model': results[best_model_name]['model'],
        'test_accuracy': results[best_model_name]['test_accuracy'],
        'cv_mean': results[best_model_name]['cv_mean'],
    }

    return {
        'results': results,
        'best_model': best_model,
        'scaler': scaler,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
    }
