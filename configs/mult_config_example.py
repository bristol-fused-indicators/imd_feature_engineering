from pathlib import Path
from itertools import product
from project_paths import paths

from imd_features.config import FeatureSetConfig, GroupConfig, ReductionMethod
from imd_features.process import create_feature_set


INPUT_PATH = paths.input_file

CRIME_COLUMNS = [
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
]

UC_COLUMNS = [
    "total_claims",
    "mean_monthly_claims",
    "%_claims_nwr",
    "%_claims_planfw",
    "%_claims_prepfw",
    "%_claims_sfw",
]

OSM_AMENITIES = [
    f"count_{group}_{dist}"
    for group in [
        "healthcare_access",
        "education_skills",
        "food_dining",
        "fast_food_takeaway",
        "transport_public",
        "essential_services",
    ]
    for dist in [500, 1000, 2000]
] + ["nearest_shop", "ratio_fast_food_takeaway_to_food_dining_1000"]

REDUCTION_METHODS = [ReductionMethod.PCA, ReductionMethod.FA]
N_COMPONENTS = [3, 5, 8]


def generate_configs() -> list[FeatureSetConfig]:
    configs = [
        FeatureSetConfig(
            name=f"grid_{method.value}_{n}",
            description=f"OSM reduced with {method.value} to {n} components",
            groups={
                "crime": GroupConfig(columns=CRIME_COLUMNS, scale=True),
                "uc": GroupConfig(columns=UC_COLUMNS, scale=True),
                "osm": GroupConfig(
                    columns=OSM_AMENITIES,
                    scale=True,
                    reduction_method=method,
                    n_components=n,
                ),
            },
        )
        for method, n in product(REDUCTION_METHODS, N_COMPONENTS)
    ]

    return configs


if __name__ == "__main__":
    configs = generate_configs()

    for config in configs:
        df, metadata = create_feature_set(INPUT_PATH, config)
        print(f"  {config.output_name}: {df.shape[1]} columns")
