from pathlib import Path
from imd_features.config import FeatureSetConfig, GroupConfig, ReductionMethod
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import (
    PCA,
    FactorAnalysis,
    FastICA,
    NMF,
    non_negative_factorization,
)
from project_paths import paths
import json
import hashlib


def process_group(
    df: pl.DataFrame, group_name: str, group_config: GroupConfig
) -> pl.DataFrame:

    data = df.to_numpy()

    if group_config.scale:
        data = apply_scaling(data)

    if group_config.reduction_method != ReductionMethod.NONE:
        data = reduce_dimensions(
            data, group_config.reduction_method, group_config.n_components
        )
        columns = [
            f"{group_name}_{group_config.reduction_method.value}_{i + 1}"
            for i in range(data.shape[1])
        ]
    else:
        columns = df.columns

    return pl.DataFrame(data, schema=columns)


def reduce_dimensions(
    data: np.ndarray, method: ReductionMethod, n_components: int
) -> np.ndarray:
    if method == ReductionMethod.PCA:
        reducer = PCA(n_components=n_components, random_state=1)

    elif method == ReductionMethod.NMF:
        if np.any(data < 0):
            raise ValueError("NMF requires non negative input")
        reducer = NMF(n_components=n_components, random_state=1, max_iter=500)

    elif method == ReductionMethod.FA:
        reducer = FactorAnalysis(
            n_components=n_components, rotation="varimax", random_state=1
        )

    elif method == ReductionMethod.ICA:
        reducer = FastICA(n_components=n_components, random_state=1, max_iter=500)

    else:
        raise ValueError(f"Unknown reduction method: {method}")

    return reducer.fit_transform(data)


def apply_scaling(data: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def compute_input_hash(path: Path) -> str:
    schema = pl.read_parquet_schema(path)
    schema_str = json.dumps(
        {name: str(dtype) for name, dtype in sorted(schema.items())},
        sort_keys=True,
    )
    return hashlib.sha256(schema_str.encode()).hexdigest()[:8]


def create_feature_set(input_data: Path, config: FeatureSetConfig) -> pl.DataFrame:

    input_df = pl.read_parquet(input_data)
    null_counts = input_df.null_count()
    if null_counts.sum_horizontal().item() > 0:
        raise ValueError("Input data contains null values, provide a clean parquet.")

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
