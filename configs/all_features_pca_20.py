"""PCA reduction to 20 components across all features."""

from pathlib import Path
from project_paths import paths
import polars as pl

from imd_features.config import FeatureSetConfig, GroupConfig, ReductionMethod
from imd_features.process import create_feature_set


INPUT_PATH = paths.input_file


def get_all_feature_columns(path: Path) -> list[str]:
    schema = pl.read_parquet_schema(path)
    return [col for col in schema.keys() if col != "lsoa_code"]


config = FeatureSetConfig(
    name="all_features_pca_20",
    description="All features in a single group, scaled and reduced to 20 PCA components",
    groups={
        "all": GroupConfig(
            columns=get_all_feature_columns(INPUT_PATH),
            scale=True,
            reduction_method=ReductionMethod.PCA,
            n_components=20,
        ),
    },
)


if __name__ == "__main__":
    df, _ = create_feature_set(INPUT_PATH, config)
    print(f"Created {config.output_name}: {df.shape[0]} rows, {df.shape[1]} columns")
