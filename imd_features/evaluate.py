import polars as pl
from imd_features.config import FeatureSetConfig
from imd_features.inspect import resolve_output_columns

from project_paths import paths
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
from scipy 


def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    model,
    feature_columns: list[str],
    group_columns: dict[str, list[str]],
) -> dict:

    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

    r2_scores = []
    rmse_scores = []
    spearman_scores = []
    importance_per_fold = []

    for train_idx, test_idx in k_fold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)

        r2_scores.append(r2_score(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        spearman_scores.append(spearmanr(y_test, y_pred).statistic) # type: ignore (checked attribute exists in src code)

        if isinstance(model_clone, RandomForestRegressor):
            importance_per_fold.append(model_clone.feature_importances_)
        elif isinstance(model_clone, Ridge):
            importance_per_fold.append(np.abs(model_clone.coef_))


            
    return {
        "r2_mean": ...,
        "r2_std": ...,
        "rmse_mean": ...,
        "rmse_std": ...,
        "spearman_mean": ...,
        "spearman_std": ...,
        "feature_importance": ...,
        "group_importance": ...,
    }


def evaluate(
    df: pl.DataFrame, config: FeatureSetConfig, target: str = "score"
) -> dict[str, dict]:

    target_df = pl.read_parquet(paths.reference / "imd_target.parquet")

    combined = df.join(target_df, on="lsoa_code", how="inner")

    feature_cols = [c for c in df.columns if c != "lsoa_code"]
    X = combined.select(feature_cols).to_numpy()
    y = combined.select(target).to_numpy().ravel()

    group_columns = resolve_output_columns(df=df, config=config)

    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    for model_name, model in models.items():
        results[model_name] = evaluate_model(X, y, model, feature_cols, group_columns)

    return results
