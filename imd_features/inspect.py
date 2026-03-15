from imd_features.config import FeatureSetConfig, ReductionMethod
import polars as pl
from matplotlib.figure import Figure


def group_summary(config: FeatureSetConfig, group_metadata: dict) -> pl.DataFrame:
    rows = []
    for group_name, group_config in config.groups.items():
        metadata = group_metadata.get(group_name, {})
        input_cols = len(group_config.columns)

        if group_config.reduction_method != ReductionMethod.NONE:
            output_cols = group_config.n_components
        else:
            output_cols = input_cols

        row = {
            "group": group_name,
            "input_features": input_cols,
            "output_features": output_cols,
            "scaled": group_config.scale,
            "reduction": group_config.reduction_method.value,
        }

        if "total_explained_variance" in metadata:
            row["explained_var"] = round(metadata["total_explained_variance"], 4)
        if "reconstruction_error" in metadata:
            row["reconstruction_err"] = round(metadata["reconstruction_error"], 4)
        if "mean_noise_variance" in metadata:
            row["mean_noise_var"] = round(metadata["mean_noise_variance"], 4)
        if "converged" in metadata:
            row["converged"] = metadata["converged"]

        rows.append(row)

    return pl.DataFrame(rows)


def correlation_within_groups(
    df: pl.DataFrame, config: FeatureSetConfig
) -> dict[str, Figure]: ...


def correlation_between_groups(
    df: pl.DataFrame, config: FeatureSetConfig
) -> Figure: ...


def correlation_full(df: pl.DataFrame) -> Figure: ...


def distribution_plot(df: pl.DataFrame, config: FeatureSetConfig) -> Figure: ...
