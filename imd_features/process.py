from pathlib import Path
from imd_features.config import FeatureSetConfig, GroupConfig, ReductionMethod
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from project_paths import paths
import json
import hashlib


def process_group(
    df: pl.DataFrame, group_name: str, group_config: GroupConfig
) -> pl.DataFrame: ...


def reduce_dimensions(
    data: np.ndarray, method: ReductionMethod, n_components: int
) -> tuple[np.ndarray, object]: ...


def apply_scaling(data: np.ndarray) -> tuple[np.ndarray, StandardScaler]: ...


def compute_input_hash(path: Path) -> str:
    schema = pl.read_parquet_schema(path)
    schema_str = json.dumps(
        {name: str(dtype) for name, dtype in sorted(schema.items())},
        sort_keys=True,
    )
    return hashlib.sha256(schema_str.encode()).hexdigest()[:8]


def create_feature_set(input_data: Path, config: FeatureSetConfig) -> pl.DataFrame:

    input_df = pl.read_parquet(input_data)
    input_hash = compute_input_hash(input_data)

    id_column = input_df.select("lsoa_code")
    group_dfs = [id_column]
    for group_name, group_config in config.groups.items():
        group_df = input_df.select(group_config.columns)
        group_df = process_group(group_df, group_name, group_config)
        group_dfs.append(group_df)

    featureset_df = pl.concat(group_dfs, how="horizontal")

    output_stem = config.output_name
    featureset_df.write_parquet(paths.output / f"{output_stem}.parquet")

    manifest = config.create_manifest_dict(input_hash)
    manifest_path = paths.output / f"{output_stem}_config.json"
    manifest_path.write_text(json.dumps(manifest))

    return featureset_df
