from sklearn.linear_model import LinearRegression, Ridge
from imd_features.config import FeatureSetConfig, GroupConfig
from project_paths import paths, project_root
import joblib
import polars as pl
from pathlib import Path

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

MODEL_2025_PATH = paths.data_models / "2025_model.joblib"
MODEL_2019_PATH = paths.data_models / "2019_model.joblib"


def train_2025_model(data_2025, target_2025):

    if MODEL_2025_PATH.exists():
        model = joblib.load(MODEL_2025_PATH)
        return model

    ...

    model = ...
    joblib.dump(model, MODEL_2025_PATH)

    return model


def train_2019_model(data_2019, target_2019):

    if MODEL_2019_PATH.exists():
        model = joblib.load(MODEL_2019_PATH)
        return model

    ...

    model = ...
    joblib.dump(model, MODEL_2019_PATH)

    return model


def predictor(): ...


def fit_models(force_retrain: bool = False) -> None: ...


def predict_quarter(
    quarterly_parquet_path: Path, snapshot_date: str
) -> pl.DataFrame: ...
