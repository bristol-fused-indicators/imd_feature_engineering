"""All features, unreduced. Baseline for comparison."""

from pathlib import Path
import polars as pl
from project_paths import paths

from imd_features.config import FeatureSetConfig, GroupConfig
from imd_features.process import create_feature_set


INPUT_PATH = paths.input_file


def get_all_feature_columns(path: Path) -> list[str]:
    schema = pl.read_parquet_schema(path)
    return [col for col in schema.keys() if col != "lsoa_code"]


config = FeatureSetConfig(
    name="all_features_unreduced",
    description="All features in a single group, no scaling or reduction",
    groups={
        "all": GroupConfig(
            columns=get_all_feature_columns(INPUT_PATH),
        ),
    },
)


if __name__ == "__main__":
    df = create_feature_set(INPUT_PATH, config)
    print(f"Created {config.output_name}: {df.shape[0]} rows, {df.shape[1]} columns")
