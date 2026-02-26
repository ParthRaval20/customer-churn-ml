from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import numpy as np


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, eval_metric='logloss', random_state=42, n_jobs=1)
    }
    return models


def train_and_compare(X_train, y_train, preprocessor):
    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"  Training {name}...", end="", flush=True)
        try:
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=3,
                scoring="f1",
                n_jobs=1
            )

            results[name] = np.mean(scores)
            print(f"  F1: {np.mean(scores):.4f}")
        except Exception as e:
            print(f"  Error: {str(e)}")
            results[name] = 0.0

    return results


def tune_best_model(model_name, X_train, y_train, preprocessor):
    """
    Hyperparameter tuning using GridSearchCV
    """

    models = get_models()
    model = models[model_name]

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Different param grids per model (industry practice)
    param_grids = {
        "Random Forest": {
            "model__n_estimators": [50, 100]
        },
        "Gradient Boosting": {
            "model__n_estimators": [50, 100]
        },
        "XGBoost": {
            "model__n_estimators": [50, 100]
        },
        "Logistic Regression": {
            "model__C": [0.1, 1]
        },
        "Decision Tree": {
            "model__max_depth": [10, 15]
        }
    }

    param_grid = param_grids.get(model_name, {})

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=1,
        verbose=0
    )

    # Fit the grid search
    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)
    print("Best CV Score:", grid.best_score_)

    return grid.best_estimator_