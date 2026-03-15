import polars as pl
from imd_features.config import FeatureSetConfig

from project_paths import paths
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


def evaluate(
    df: pl.DataFrame, config: FeatureSetConfig, target: str = "score"
) -> dict[str, dict]:

    target_df = pl.read_parquet(paths.reference / "imd_target.parquet")

    combined = df.join(target_df, on="lsoa_code", how="inner")

    feature_cols = [c for c in df.columns if c != "lsoa_code"]
    X = combined.select(feature_cols).to_numpy()
    y = combined.select(target).to_numpy().ravel()

    ...
    group_columns = ...  # todo

    ...

    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    for model_name, model in models.items():
        # evaluate each model, get some results
        # update results for each model name
        ...

    #! placeholders to make lsp shut up
    results = {}
    return results
