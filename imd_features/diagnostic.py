from imd_features.config import FeatureSetConfig, ReductionMethod
import polars as pl
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


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


def _correlation_matrix(df: pl.DataFrame) -> np.ndarray:
    return np.corrcoef(df.to_numpy(), rowvar=False)


def correlation_within_groups(
    df: pl.DataFrame, config: FeatureSetConfig
) -> dict[str, Figure]:
    group_columns = _resolve_output_columns(df, config)
    figures = {}

    for group_name, cols in group_columns.items():
        if len(cols) < 2:
            continue

        corr = _correlation_matrix(df.select(cols))

        fig, ax = plt.subplots(
            figsize=(max(6, len(cols) * 0.4), max(5, len(cols) * 0.35))
        )
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(cols, fontsize=7)
        ax.set_title(f"{group_name} — within-group correlation")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()

        figures[group_name] = fig

    return figures


def correlation_between_groups(df: pl.DataFrame, config: FeatureSetConfig) -> Figure:
    group_columns = _resolve_output_columns(df, config)
    group_names = list(group_columns.keys())
    n = len(group_names)

    mean_abs_corr = np.zeros((n, n))
    for i, name_a in enumerate(group_names):
        for j, name_b in enumerate(group_names):
            if i == j:
                mean_abs_corr[i, j] = 1.0
                continue

            cols_a = group_columns[name_a]
            cols_b = group_columns[name_b]
            if not cols_a or not cols_b:
                mean_abs_corr[i, j] = 0.0
                continue

            combined = df.select(cols_a + cols_b)
            full_corr = _correlation_matrix(combined)

            cross_block = full_corr[: len(cols_a), len(cols_a) :]
            mean_abs_corr[i, j] = np.abs(cross_block).mean()

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n)))
    im = ax.imshow(mean_abs_corr, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(group_names, rotation=45, ha="right")
    ax.set_yticklabels(group_names)
    ax.set_title("Mean absolute correlation between groups")
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{mean_abs_corr[i, j]:.2f}", ha="center", va="center", fontsize=9
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    return fig


def correlation_full(df: pl.DataFrame) -> Figure:
    cols = [c for c in df.columns if c != "lsoa_code"]
    corr = _correlation_matrix(df.select(cols))

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_title("Full feature correlation matrix")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    return fig


def distribution_plot(df: pl.DataFrame, config: FeatureSetConfig) -> Figure:
    group_columns = resolve_output_columns(df, config)
    group_names = [name for name, cols in group_columns.items() if cols]
    n = len(group_names)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, group_name in zip(axes, group_names):
        cols = group_columns[group_name]
        data = df.select(cols).to_numpy()
        ax.boxplot(data, vert=True)
        ax.set_title(group_name, fontsize=10)
        ax.set_xticklabels(cols, rotation=90, fontsize=6)

    fig.suptitle("Feature distributions by group", fontsize=12)
    fig.tight_layout()

    return fig


# for helping with mapping original columns to reduced columns
# should revisit if strong contract is established with imd pipeline with manifest file
# or emit this info from the process funcs
def resolve_output_columns(
    df: pl.DataFrame, config: FeatureSetConfig
) -> dict[str, list[str]]:
    group_columns = {}
    df_columns = set(df.columns) - {"lsoa_code"}

    for group_name, group_config in config.groups.items():
        if group_config.reduction_method != ReductionMethod.NONE:
            prefix = f"{group_name}_{group_config.reduction_method.value}_"
            cols = [c for c in df.columns if c.startswith(prefix)]
        else:
            cols = [c for c in group_config.columns if c in df_columns]

        group_columns[group_name] = cols

    return group_columns
