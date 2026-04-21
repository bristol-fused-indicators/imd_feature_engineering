from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from imd_features.config import FeatureSetConfig, GroupConfig
from imd_features.process import create_feature_set
from project_paths import paths, project_root
import joblib
import polars as pl
from pathlib import Path
from uuid import uuid4
import os
from datetime import date
from dateutil.relativedelta import relativedelta
from icecream import ic

input_dir = project_root / "data" / "input"

ANCHOR_FILE_PATHS = {
    "2019": project_root / "data" / "input" / "combined_data_2019-09-01.parquet",
    "2025": project_root / "data" / "input" / "combined_data_2025-10-01.parquet",
}

ANCHOR_TARGET_PATHS = {
    "2025": project_root / "data" / "input" / "2025_imd_target.parquet",
    "2019": project_root / "data" / "input" / "2019_imd_target.parquet",
}

CONFIG_2025 = FeatureSetConfig(
    name="rate_features_test",
    description="Rate-based features for comparison against share-based design",
    groups={
        "index": GroupConfig(columns=["snapshot_date"]),
        "crime": GroupConfig(
            columns=[
                "crime_rate_per_1000",
                "violent_crime_rate",
                "burglary_rate",
                "drugs_rate",
                "resolution_rate",
            ]
        ),
        "uc": GroupConfig(
            columns=[
                "uc_claim_rate",
                "uc_nwr_rate",
                "%_claims_planfw",
                "%_claims_sfw",
            ]
        ),
        "land_registry": GroupConfig(
            columns=[
                "lsoa_mean_price",
                "lsoa_median_price",
                "lsoa_price_inequality",
                "transactions_per_capita",
                "flats_proportion",
                "terraced_proportion",
                "detached_proportion",
                "freehold_proportion",
            ]
        ),
        "osm_landuse": GroupConfig(
            columns=[
                "landuse_residential_0",
                "landuse_industrial_0",
                "landuse_commercial_0",
                "streetlit_percentage",
            ]
        ),
        "demographics": GroupConfig(
            columns=[
                "youth_share",
                "elderly_share",
            ]
        ),
    },
)

CONFIG_2019 = FeatureSetConfig(
    name="rate_features_2019_osm",
    description="Medium rates plus OSM nearest distance and buffered counts",
    groups={
        "index": GroupConfig(columns=["snapshot_date"]),
        "crime": GroupConfig(
            columns=[
                "crime_rate_per_1000",
                "violent_crime_rate",
                "burglary_rate",
                "drugs_rate",
                "resolution_rate",
            ]
        ),
        "uc": GroupConfig(
            columns=[
                "uc_claim_rate",
                "uc_nwr_rate",
                "%_claims_planfw",
                "%_claims_sfw",
            ]
        ),
        "land_registry": GroupConfig(
            columns=[
                "lsoa_mean_price",
                "lsoa_median_price",
                "lsoa_price_inequality",
                "transactions_per_capita",
                "flats_proportion",
                "terraced_proportion",
                "detached_proportion",
                "freehold_proportion",
            ]
        ),
        "osm_landuse": GroupConfig(
            columns=[
                "landuse_residential_0",
                "landuse_industrial_0",
                "landuse_commercial_0",
                "streetlit_percentage",
            ]
        ),
        "osm_nearest": GroupConfig(
            columns=[
                "nearest_pharmacy",
                "nearest_hospital",
                "nearest_school",
                "nearest_kindergarten",
                "nearest_college",
                "nearest_university",
                "nearest_gambling",
            ]
        ),
        "osm_counts": GroupConfig(
            columns=[
                "count_healthcare_access_500",
                "count_healthcare_access_1000",
                "count_education_skills_500",
                "count_education_skills_1000",
                "count_essential_services_500",
                "count_essential_services_1000",
                "count_fast_food_takeaway_500",
                "count_fast_food_takeaway_1000",
                "count_alcohol_gambling_500",
                "count_alcohol_gambling_1000",
            ]
        ),
        "demographics": GroupConfig(
            columns=[
                "youth_share",
                "elderly_share",
            ]
        ),
    },
)

MODEL_2025_PATH = paths.models / "2025_model.joblib"
MODEL_2019_PATH = paths.models / "2019_model.joblib"


def train_2025_model():

    scaler_path = MODEL_2025_PATH.parent / "2025_fitted_scaler.joblib"

    if MODEL_2025_PATH.exists() and scaler_path.exists():
        model = joblib.load(MODEL_2025_PATH)
        scaler = joblib.load(scaler_path)
        return model, scaler

    input_data = pl.read_parquet(ANCHOR_FILE_PATHS["2025"])
    targets = pl.read_parquet(ANCHOR_TARGET_PATHS["2025"])

    rates_df = create_rate_features(input_data)

    temp_path = project_root / f"temp_{uuid4()}.parquet"
    rates_df.write_parquet(temp_path)
    features_df, *_ = create_feature_set(temp_path, config=CONFIG_2025)
    if temp_path.exists():
        os.remove(temp_path)

    feature_cols = [
        column
        for column in features_df.columns
        if column not in {"lsoa_code", "snapshot_date"}
    ]

    combined = features_df.join(targets, on="lsoa_code", how="inner")
    x_unscaled = combined.select(feature_cols).to_numpy()
    y = combined.get_column("score").to_numpy()

    scaler = StandardScaler().fit(X=x_unscaled)
    X = scaler.transform(X=x_unscaled)

    model = LinearRegression().fit(X, y)
    joblib.dump(model, MODEL_2025_PATH)
    joblib.dump(scaler, scaler_path)

    return model, scaler


def train_2019_model():

    scaler_path = MODEL_2019_PATH.parent / "2019_fitted_scaler.joblib"

    if MODEL_2019_PATH.exists() and scaler_path.exists():
        model = joblib.load(MODEL_2019_PATH)
        scaler = joblib.load(scaler_path)
        return model, scaler

    input_data = pl.read_parquet(ANCHOR_FILE_PATHS["2019"])
    targets = pl.read_parquet(ANCHOR_TARGET_PATHS["2019"])

    rates_df = create_rate_features(input_data)

    temp_path = project_root / f"temp_{uuid4()}.parquet"
    rates_df.write_parquet(temp_path)
    features_df, *_ = create_feature_set(temp_path, config=CONFIG_2019)
    if temp_path.exists():
        os.remove(temp_path)

    feature_cols = [
        column
        for column in features_df.columns
        if column not in {"lsoa_code", "snapshot_date"}
    ]

    combined = features_df.join(targets, on="lsoa_code", how="inner")
    x_unscaled = combined.select(feature_cols).to_numpy()
    y = combined.get_column("score").to_numpy()

    scaler = StandardScaler().fit(X=x_unscaled)
    X = scaler.transform(X=x_unscaled)

    model = Ridge(alpha=25.3).fit(X, y)
    joblib.dump(model, MODEL_2019_PATH)
    joblib.dump(scaler, MODEL_2019_PATH.parent / "2019_fitted_scaler.joblib")

    return model, scaler


def create_rate_features(raw: pl.DataFrame) -> pl.DataFrame:
    return raw.with_columns(
        (pl.col("total_crimes") / pl.col("lsoa_population") * 1000).alias(
            "crime_rate_per_1000"
        ),
        (pl.col("violent-crime") / pl.col("lsoa_population") * 1000).alias(
            "violent_crime_rate"
        ),
        (pl.col("burglary") / pl.col("lsoa_population") * 1000).alias("burglary_rate"),
        (pl.col("drugs") / pl.col("lsoa_population") * 1000).alias("drugs_rate"),
        (pl.col("total_claims") / pl.col("working_age_population")).alias(
            "uc_claim_rate"
        ),
        (pl.col("total_nwr_claims") / pl.col("working_age_population")).alias(
            "uc_nwr_rate"
        ),
        (pl.col("total_transactions") / pl.col("lsoa_population") * 1000).alias(
            "transactions_per_capita"
        ),
        (pl.col("aged_under_15") / pl.col("lsoa_population")).alias("youth_share"),
        (pl.col("pension_age_population") / pl.col("lsoa_population")).alias(
            "elderly_share"
        ),
    )


def predictor(X_2025, X_2019, snapshot_date: str, model_25, model_19):
    anchor_2025 = date(2025, 10, 1)
    anchor_2019 = date(2019, 9, 1)

    ref = date.fromisoformat(snapshot_date)

    ic(anchor_2025, anchor_2019, ref)

    dist_from_25_anchor = abs((anchor_2025 - ref).days)
    dist_from_19_anchor = abs((anchor_2019 - ref).days)

    ic(dist_from_19_anchor, dist_from_25_anchor)

    # larger when closer to 2025, smaller when further away
    w_25 = 1 - (dist_from_25_anchor / (dist_from_19_anchor + dist_from_25_anchor))
    w_19 = 1 - w_25

    ic(w_19, w_25)

    model_25_score = model_25.predict(X_2025)
    model_19_score = model_19.predict(X_2019)

    score = w_25 * model_25_score + w_19 * model_19_score

    return score


def fit_models(force_retrain: bool = False):
    if force_retrain:
        for path in [
            MODEL_2019_PATH,
            MODEL_2025_PATH,
            MODEL_2019_PATH.parent / "2019_fitted_scaler.joblib",
            MODEL_2025_PATH.parent / "2025_fitted_scaler.joblib",
        ]:
            if path.exists():
                os.remove(path)

    model_2019, scaler_2019 = train_2019_model()
    model_2025, scaler_2025 = train_2025_model()
    return (model_2019, scaler_2019), (model_2025, scaler_2025)


def predict_quarter(
    quarterly_parquet_path: Path, snapshot_date: str
) -> pl.DataFrame: ...


if __name__ == "__main__":
    predictor(None, None, "2024-03-01", None, None)
