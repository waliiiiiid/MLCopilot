import os
import warnings
import joblib
import pandas as pd
from typing import Dict, List, Any
from langchain_core.tools import tool
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

MODELS = {
    "RandomForestClassifier": RandomForestClassifier,
    "RandomForestRegressor": RandomForestRegressor,
    "LogisticRegression": LogisticRegression,
    "LinearRegression": LinearRegression,
    "XGBClassifier": XGBClassifier,
    "XGBRegressor": XGBRegressor,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "SVR": SVR,
}

MAX_GRID_COMBOS = 30


@tool
def optimize_model(
    model_name: str,
    file_path: str,
    target_column: str,
    scoring: str,
    parameter_grid: Dict[str, List[Any]],
) -> dict:
    """Tune a model with GridSearchCV, refit it on the full dataset, save it, and return scores.

    model_name: one of RandomForestClassifier, RandomForestRegressor, LogisticRegression,
                LinearRegression, XGBClassifier, XGBRegressor, GradientBoostingClassifier, SVR.
    file_path: path to the cleaned CSV file.
    target_column: name of the target column.
    scoring: sklearn scoring string, e.g. 'accuracy', 'f1_weighted', 'r2',
             'neg_root_mean_squared_error'.
    parameter_grid: e.g. {'n_estimators': [100, 200], 'max_depth': [5, 10]}.
                    Keep it small (at most 30 combinations).

    cv_score is the mean 5-fold cross-validation score on the 80% training split.
    test_score was measured on a held-out 20% split. The saved model is refit on
    100% of the data using the best parameters.
    """
    try:
        if model_name not in MODELS:
            return {"error": f"Unknown model '{model_name}'. Choose from {list(MODELS)}"}

        n_combos = len(ParameterGrid(parameter_grid))
        if n_combos > MAX_GRID_COMBOS:
            return {"error": f"Grid has {n_combos} combinations; reduce it to at most {MAX_GRID_COMBOS}."}

        df = pd.read_csv(file_path)
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found. Columns: {list(df.columns)}"}

        X, y = df.drop(columns=[target_column]), df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 1. tune on the 80% split
        gs = GridSearchCV(MODELS[model_name](), parameter_grid, scoring=scoring, cv=5)
        gs.fit(X_train, y_train)


        # 2. score on the held-out 20%
        test_score = float(gs.score(X_test, y_test))

        # 3. refit on the full dataset with the best parameters and save
        final_model = MODELS[model_name](**gs.best_params_).fit(X, y)
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{model_name}.joblib"
        joblib.dump(final_model, model_path)

        return {
            "model_name": model_name,
            "best_parameters": gs.best_params_,
            "cv_score": round(float(gs.best_score_), 4),
            "test_score": round(test_score, 4),
            "model_path": model_path,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}