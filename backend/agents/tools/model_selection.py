from langchain_core.tools import tool
import pandas as pd
import sklearn 
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,RandomForestRegressor
from sklearn.svm import SVC,SVR
from xgboost import XGBClassifier,XGBRegressor

# Dataset lives here, not in tool arguments
DATA = {"df": None}


@tool
def select_model(file_path: str, task: str, target_column: str, metric: str = "") -> dict:
    """
    Compare candidate ML models using 5-fold cross-validation.

    Args:
        file_path: Path to the preprocessed CSV file.
        task: "classification" or "regression".
        target_column: Name of the target column.
        metric: Optional sklearn scoring string (e.g. "f1_macro", "accuracy", "r2").
                Defaults to accuracy for classification and r2 for regression.

    Returns:
        The best model name and mean CV scores for all candidate models.
    """
    df = pd.read_csv(file_path)

    if target_column not in df.columns:
        return {"error": f"Column '{target_column}' not found. Columns: {list(df.columns)}"}

    x = df.drop(columns=[target_column])
    y = df[target_column]

    if task == "classification":
        models = [
            LogisticRegression(max_iter=1000),
            RandomForestClassifier(random_state=42),
            XGBClassifier(random_state=42),
            GradientBoostingClassifier(random_state=42),
        ]
        scoring = metric or "accuracy"
    elif task == "regression":
        models = [
            LinearRegression(),
            RandomForestRegressor(random_state=42),
            XGBRegressor(random_state=42),
            SVR(),
        ]
        scoring = metric or "r2"
    else:
        return {"error": "task must be 'classification' or 'regression'"}

    results = {}
    for model in models:
        name = type(model).__name__
        try:
            scores = cross_val_score(model, x, y, cv=5, scoring=scoring)
            results[name] = round(float(scores.mean()), 4)
        except Exception as e:
            results[name] = f"error: {e}"

    valid = {k: v for k, v in results.items() if isinstance(v, float)}
    if not valid:
        return {"error": "All models failed", "details": results}

    best_model = max(valid, key=valid.get)
    return {
        "best_model": best_model,
        "metric": scoring,
        "best_score": valid[best_model],
        "scores": results,
    }