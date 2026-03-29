import polars as pl
from imd_features.config import FeatureSetConfig
from imd_features.diagnostic import resolve_output_columns
from imd_features.spatial_utils import fetch_spatial_support_data

from project_paths import paths
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
# from scipy

from icecream import ic


def slx_cv(X: np.ndarray, y: np.ndarray, model_spec: dict) -> dict:
    
    r2_scores = []
    rmse_scores = []
    spearman_scores = []
    importance_per_fold = []

    W, groups = fetch_spatial_support_data()

    gkf = GroupKFold(n_splits=5)

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        WX_train = W[np.ix_(train_idx, train_idx)] @ X_train
        WX_test  = W[np.ix_(test_idx, test_idx)]  @ X_test

        X_train_slx = np.hstack([X_train, WX_train])
        X_test_slx = np.hstack([X_test, WX_test])

        k = X_train.shape[1] # index for cutoff on direct vs neighbor features

        model = model_spec['model'].fit(X_train_slx, y_train)
        y_pred = model.predict(X_test_slx)

        beta = model.coef_[:k]      # direct effects
        theta = model.coef_[k:]     # spatial lag effects

        r2_scores.append(r2_score(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        spearman_scores.append(spearmanr(y_test, y_pred).statistic) 
        importance_per_fold.append(np.abs(beta) + np.abs(theta))  # simple way to combine direct and spatial effects for importance
    
    print(r2_scores)

    return r2_scores, rmse_scores, spearman_scores, importance_per_fold  

def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    model,
    feature_columns: list[str],
    group_columns: dict[str, list[str]],
) -> dict:


    if isinstance(model, dict) and model.get('model_type') == 'SLX':
        r2_scores, rmse_scores, spearman_scores, importance_per_fold = slx_cv(X=X, y=y, model_spec=model)
    
    else:
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
            spearman_scores.append(spearmanr(y_test, y_pred).statistic)  # type: ignore (checked attribute exists in src code)

            if isinstance(model_clone, RandomForestRegressor):
                importance_per_fold.append(model_clone.feature_importances_)
            elif isinstance(model_clone, Ridge):
                importance_per_fold.append(np.abs(model_clone.coef_))

    importance_mean = np.mean(importance_per_fold, axis=0)
    importance_std = np.std(importance_per_fold, axis=0)

    col_to_group = {}
    for group_name, cols in group_columns.items():
        for col in cols:
            col_to_group[col] = group_name

    feature_importance = [
        {
            "feature": col,
            "group": col_to_group.get(col, "unknown"),
            "importance_mean": float(importance_mean[i]),
            "importance_std": float(importance_std[i]),
        }
        for i, col in enumerate(feature_columns)
    ]

    group_importance = []
    for group_name, cols in group_columns.items():
        indices = [i for i, c in enumerate(feature_columns) if c in set(cols)]
        if not indices:
            continue
        group_imp = importance_mean[indices]
        group_importance.append(
            {
                "group": group_name,
                "total_importance": float(group_imp.sum()),
                "mean_importance": float(group_imp.mean()),
                "n_features": len(indices),
            }
        )

    return {
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "spearman_mean": float(np.mean(spearman_scores)),
        "spearman_std": float(np.std(spearman_scores)),
        "feature_importance": feature_importance,
        "group_importance": group_importance,
    }


def evaluate(
    df: pl.DataFrame, config: FeatureSetConfig, target: str = "score"
) -> dict[str, dict]:

    target_df = pl.read_parquet(paths.reference)
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
        "slx": {'model_type': 'SLX',
                'reg_type': 'Ridge',
                'model': Ridge(alpha=1.0)},  
    }

    results = {}
    for model_name, model in models.items():
        results[model_name] = evaluate_model(X, y, model, feature_cols, group_columns)

    return results


if __name__ == "__main__":
    data = pl.read_parquet(paths.input_file)
    config_path = paths.output / "all_features_pca_20_1bd1256f_config.json"
    config = FeatureSetConfig.model_validate_json(config_path.read_text())
    evaluate(data, config)
