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


def correlation_within_groups(
    df: pl.DataFrame, config: FeatureSetConfig
) -> dict[str, Figure]:
    group_columns = _resolve_output_columns(df, config)
    figures = {}

    for group_name, cols in group_columns.items():
        if len(cols) < 2:
            continue

        corr = df.select(cols).to_pandas().corr()

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

    mean_corr = np.zeros((n, n))
    for i, name_a in enumerate(group_names):
        for j, name_b in enumerate(group_names):
            if i == j:
                mean_corr[i, j] = 1.0
            else:
                cols_a = group_columns[name_a]
                cols_b = group_columns[name_b]
                if not cols_a or not cols_b:
                    mean_corr[i, j] = 0.0
                    continue
                cross = np.abs(
                    df.select(cols_a)
                    .to_pandas()
                    .corrwith(df.select(cols_b).to_pandas())
                )
                mean_corr[i, j] = cross.mean()

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n)))
    im = ax.imshow(mean_corr, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(group_names, rotation=45, ha="right")
    ax.set_yticklabels(group_names)
    ax.set_title("Mean absolute correlation between groups")
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{mean_corr[i, j]:.2f}", ha="center", va="center", fontsize=9
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    return fig


def correlation_full(df: pl.DataFrame) -> Figure:
    cols = [c for c in df.columns if c != "lsoa_code"]
    corr = df.select(cols).to_pandas().corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_title("Full feature correlation matrix")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    return fig


def distribution_plot(df: pl.DataFrame, config: FeatureSetConfig) -> Figure:
    group_columns = _resolve_output_columns(df, config)
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
def _resolve_output_columns(
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
