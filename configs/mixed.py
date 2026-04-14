"""Per-group reduction strategies based on group characteristics."""

from project_paths import paths
from imd_features.config import FeatureSetConfig, GroupConfig, ReductionMethod
from imd_features.process import create_feature_set


INPUT_PATH = paths.input_file


config = FeatureSetConfig(
    name="mixed_reduction",
    description=(
        "Large correlated groups reduced, small interpretable groups kept intact. "
        "OSM amenities PCA(15), connectivity FA(4), remaining groups scaled only."
    ),
    groups={
        "crime": GroupConfig(
            columns=[
                "violent-crime",
                "burglary",
                "anti-social-behaviour",
                "shoplifting",
                "criminal-damage-arson",
                "drugs",
                "robbery",
                "vehicle-crime",
                "other-theft",
                "public-order",
                "bicycle-theft",
                "other-crime",
                "theft-from-the-person",
                "total_crimes",
                "resolution_rate",
            ],
            scale=True,
        ),
        "universal_credit": GroupConfig(
            columns=[
                "total_claims",
                "mean_monthly_claims",
                "total_nwr_claims",
                "total_planfw_claims",
                "total_prepfw_claims",
                "total_sfw_claims",
                "%_claims_nwr",
                "%_claims_planfw",
                "%_claims_prepfw",
                "%_claims_sfw",
            ],
            scale=True,
        ),
        "connectivity": GroupConfig(
            columns=[
                "Employment (walking)",
                "Education (walking)",
                "Healthcare (walking)",
                "Leisure and Community (walking)",
                "Shopping (walking)",
                "Residential (walking)",
                "Overall (walking)",
                "Employment (cycling)",
                "Education (cycling)",
                "Healthcare (cycling)",
                "Leisure and Community (cycling)",
                "Shopping (cycling)",
                "Residential (cycling)",
                "Overall (cycling)",
                "Business (public transport)",
                "Education (public transport)",
                "Healthcare (public transport)",
                "Leisure and Community (public transport)",
                "Shopping (public transport)",
                "Residential (public transport)",
                "Overall (public transport)",
                "Employment (driving)",
                "Education (driving)",
                "Healthcare (driving)",
                "Leisure and Community (driving)",
                "Shopping (driving)",
                "Residential (driving)",
                "Overall (driving)",
                "Employment (overall)",
                "Education (overall)",
                "Healthcare (overall)",
                "Leisure and Community (overall)",
                "Shopping (overall)",
                "Residential (overall)",
                "Overall",
            ],
            scale=True,
            reduction_method=ReductionMethod.FA,
            n_components=4,
        ),
        "land_registry": GroupConfig(
            columns=[
                "lsoa_mean_price",
                "lsoa_max_price",
                "T_mean_price",
                "F_mean_price",
                "S_mean_price",
                "D_mean_price",
                "O_mean_price",
                "total_transactions",
                "T_count_transactions",
                "F_count_transactions",
                "S_count_transactions",
                "D_count_transactions",
                "O_count_transactions",
            ],
            scale=True,
        ),
        "osm_amenities": GroupConfig(
            columns=[
                f"count_{group}_{dist}"
                for group in [
                    "healthcare_access",
                    "education_skills",
                    "financial_services",
                    "food_dining",
                    "fast_food_takeaway",
                    "alcohol_gambling",
                    "community_social",
                    "cultural_entertainment",
                    "public_services_civic",
                    "transport_car",
                    "transport_public",
                    "sustainable_transport",
                    "childcare_early_years",
                    "elderly_care",
                    "public_amenities_environment",
                    "waste_sanitation",
                    "religious_spiritual",
                    "higher_education_research",
                    "professional_services",
                    "retail_commerce",
                    "essential_services",
                    "social_support",
                ]
                for dist in [0, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 5000]
            ]
            + ["nearest_shop", "ratio_fast_food_takeaway_to_food_dining_1000"],
            scale=True,
            reduction_method=ReductionMethod.PCA,
            n_components=15,
        ),
        "osm_landuse": GroupConfig(
            columns=[
                col
                for col in [
                    "landuse_allotments_0",
                    "landuse_brownfield_0",
                    "landuse_cemetery_0",
                    "landuse_commercial_0",
                    "landuse_conservation_0",
                    "landuse_construction_0",
                    "landuse_depot_0",
                    "landuse_education_0",
                    "landuse_farmland_0",
                    "landuse_farmyard_0",
                    "landuse_flowerbed_0",
                    "landuse_garages_0",
                    "landuse_grass_0",
                    "landuse_greenfield_0",
                    "landuse_greenhouse_horticulture_0",
                    "landuse_industrial_0",
                    "landuse_landfill_0",
                    "landuse_meadow_0",
                    "landuse_military_0",
                    "landuse_orchard_0",
                    "landuse_plant_nursery_0",
                    "landuse_railway_0",
                    "landuse_recreation_ground_0",
                    "landuse_religious_0",
                    "landuse_residential_0",
                    "landuse_retail_0",
                    "landuse_storage_0",
                    "landuse_terminal_0",
                    "streetlit_percentage",
                ]
            ],
            scale=True,
        ),
        "population": GroupConfig(
            columns=[
                "lsoa_population",
                "aged_under_15",
                "working_age_population",
                "pension_age_population",
            ],
            scale=True,
        ),
    },
)


if __name__ == "__main__":
    df, *_ = create_feature_set(INPUT_PATH, config)
    print(f"Created {config.output_name}: {df.shape[0]} rows, {df.shape[1]} columns")
