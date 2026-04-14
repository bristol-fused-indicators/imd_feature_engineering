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
    df: pl.DataFrame,
    group_name: str,
    group_config: GroupConfig,
    scaler: StandardScaler | None = None,
) -> tuple[pl.DataFrame, dict, StandardScaler | None]:

    metadata = {}
    data = df.to_numpy()

    if group_config.scale:
        data, scaler = apply_scaling(data, scaler)

    if group_config.reduction_method != ReductionMethod.NONE:
        data, reduction_metadata = reduce_dimensions(
            data, group_config.reduction_method, group_config.n_components
        )
        metadata.update(reduction_metadata)
        columns = [
            f"{group_name}_{group_config.reduction_method.value}_{i + 1}"
            for i in range(data.shape[1])
        ]
    else:
        columns = df.columns

    return pl.DataFrame(data, schema=columns), metadata, scaler


def reduce_dimensions(
    data: np.ndarray, method: ReductionMethod, n_components: int
) -> tuple[np.ndarray, dict]:

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

    reduced = reducer.fit_transform(data)

    metadata = {}
    if isinstance(reducer, PCA):
        metadata["explained_variance_ratio"] = (
            reducer.explained_variance_ratio_.tolist()
        )
        metadata["total_explained_variance"] = sum(reducer.explained_variance_ratio_)
    elif isinstance(reducer, NMF):
        metadata["reconstruction_error"] = float(reducer.reconstruction_err_)
    elif isinstance(reducer, FactorAnalysis):
        metadata["noise_variance"] = reducer.noise_variance_.tolist()
        metadata["mean_noise_variance"] = float(np.mean(reducer.noise_variance_))
        metadata["log_likelihood"] = float(reducer.loglike_[-1])
    elif isinstance(reducer, FastICA):
        metadata["n_iterations"] = int(reducer.n_iter_)
        metadata["converged"] = reducer.n_iter_ < 500

    return reduced, metadata


def apply_scaling(
    data: np.ndarray, scaler: StandardScaler | None = None
) -> tuple[np.ndarray, StandardScaler]:
    if scaler is None:
        scaler = StandardScaler()
        return scaler.fit_transform(data), scaler
    return scaler.transform(data), scaler


def compute_input_hash(path: Path) -> str:
    schema = pl.read_parquet_schema(path)
    schema_str = json.dumps(
        {name: str(dtype) for name, dtype in sorted(schema.items())},
        sort_keys=True,
    )
    return hashlib.sha256(schema_str.encode()).hexdigest()[:8]


def create_feature_set(
    input_data: Path,
    config: FeatureSetConfig,
    fitted_scalers: dict[str, StandardScaler] | None = None,
) -> tuple[pl.DataFrame, dict, dict[str, StandardScaler | None]]:
    fitted_scalers = fitted_scalers or {}  # replace None so dict methods work

    input_df = pl.read_parquet(input_data)
    null_counts = input_df.null_count()
    if null_counts.sum_horizontal().item() > 0:
        raise ValueError("Input data contains null values, provide a clean parquet.")

    input_hash = compute_input_hash(input_data)

    id_column = input_df.select("lsoa_code")
    group_dfs, group_metadata = [id_column], {}
    output_fitted_scalers = {}
    for group_name, group_config in config.groups.items():
        group_df = input_df.select(group_config.columns)
        scaler = fitted_scalers.get(group_name)
        group_df, metadata, scaler = process_group(
            group_df, group_name, group_config, scaler
        )
        group_dfs.append(group_df)
        group_metadata[group_name] = metadata
        output_fitted_scalers[group_name] = scaler

    featureset_df = pl.concat(group_dfs, how="horizontal")

    output_stem = config.output_name
    featureset_df.write_parquet(paths.output / f"{output_stem}.parquet")

    manifest = config.create_manifest_dict(input_hash)
    manifest_path = paths.output / f"{output_stem}_config.json"
    manifest_path.write_text(json.dumps(manifest))

    return featureset_df, group_metadata, output_fitted_scalers
