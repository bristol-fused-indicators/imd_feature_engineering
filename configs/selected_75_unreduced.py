"""Selected features, no reduction. Demonstrates manual feature selection."""

from pathlib import Path
from project_paths import paths
from imd_features.config import FeatureSetConfig, GroupConfig
from imd_features.process import create_feature_set


INPUT_PATH = paths.input_file


config = FeatureSetConfig(
    name="selected_75_unreduced",
    description="75 selected features across all sources, no reduction",
    groups={
        "crime": GroupConfig(
            columns=[
                "violent-crime",
                "burglary",
                "anti-social-behaviour",
                "shoplifting",
                "criminal-damage-arson",
                "drugs",
                "total_crimes",
                "resolution_rate",
            ],
        ),
        "universal_credit": GroupConfig(
            columns=[
                "total_claims",
                "mean_monthly_claims",
                "%_claims_nwr",
                "%_claims_planfw",
                "%_claims_prepfw",
                "%_claims_sfw",
            ],
        ),
        "connectivity": GroupConfig(
            columns=[
                "Overall (walking)",
                "Overall (cycling)",
                "Overall (public transport)",
                "Overall (driving)",
                "Overall",
                "Employment (walking)",
                "Education (walking)",
                "Healthcare (walking)",
                "Shopping (walking)",
                "Education (public transport)",
                "Healthcare (public transport)",
            ],
        ),
        "land_registry": GroupConfig(
            columns=[
                "lsoa_mean_price",
                "lsoa_max_price",
                "total_transactions",
                "T_mean_price",
                "F_mean_price",
                "S_mean_price",
                "D_mean_price",
            ],
        ),
        "osm_amenities": GroupConfig(
            columns=[
                "count_healthcare_access_500",
                "count_healthcare_access_1000",
                "count_education_skills_500",
                "count_education_skills_1000",
                "count_food_dining_500",
                "count_food_dining_1000",
                "count_fast_food_takeaway_500",
                "count_fast_food_takeaway_1000",
                "count_alcohol_gambling_500",
                "count_alcohol_gambling_1000",
                "count_transport_public_500",
                "count_transport_public_1000",
                "count_essential_services_500",
                "count_essential_services_1000",
                "count_community_social_500",
                "count_community_social_1000",
                "count_financial_services_500",
                "count_financial_services_1000",
                "count_childcare_early_years_500",
                "count_childcare_early_years_1000",
                "nearest_shop",
                "ratio_fast_food_takeaway_to_food_dining_1000",
            ],
        ),
        "osm_landuse": GroupConfig(
            columns=[
                "landuse_residential_0",
                "landuse_commercial_0",
                "landuse_industrial_0",
                "landuse_grass_0",
                "landuse_recreation_ground_0",
                "landuse_education_0",
                "streetlit_percentage",
            ],
        ),
        "population": GroupConfig(
            columns=[
                "lsoa_population",
                "aged_under_15",
                "working_age_population",
                "pension_age_population",
            ],
        ),
    },
)


if __name__ == "__main__":
    df, *_ = create_feature_set(INPUT_PATH, config)
    print(f"Created {config.output_name}: {df.shape[0]} rows, {df.shape[1]} columns")
