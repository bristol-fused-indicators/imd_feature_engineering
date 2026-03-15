from pathlib import Path
from imd_features.config import FeatureSetConfig, GroupConfig, ReductionMethod
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler


def process_group(
    df: pl.DataFrame, group_name: str, group_config: GroupConfig
) -> pl.DataFrame: ...


def reduce_dimensions(
    data: np.ndarray, method: ReductionMethod, n_components: int
) -> tuple[np.ndarray, object]: ...


def apply_scaling(data: np.ndarray) -> tuple[np.ndarray, StandardScaler]: ...


def create_feature_set(input_data: Path, config: FeatureSetConfig) -> pl.DataFrame:

    input_df = pl.read_parquet(input_data)

    group_dfs = []
    for group, group_config in config.groups.items():
        group_df = input_df.select(*group_config.columns)
        group_df = process_group(group_df, group, group_config)
        group_dfs.append(group_df)

    featureset_df = pl.concat(group_dfs, how="horizontal")

    return featureset_df
