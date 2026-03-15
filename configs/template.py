"""Template for creating feature set configurations.

Copy this file, rename it, and edit the groups to define your experiment.

Usage:
    uv run python configs/my_experiment.py
"""

from pathlib import Path
from project_paths import paths

from imd_features.config import FeatureSetConfig, GroupConfig, ReductionMethod
from imd_features.process import create_feature_set


# Define your feature set configuration
config = FeatureSetConfig(
    name="my_experiment",
    description="Describe what this config is testing",
    groups={
        # Each group is a named set of columns with optional scaling and reduction.
        # Groups cannot share columns.
        #
        # Minimal group — just select columns, no transformation:
        # "crime": GroupConfig(
        #     columns=["robbery", "burglary", "violent-crime"],
        # ),
        #
        # Scaled group — standardise before downstream use:
        # "connectivity": GroupConfig(
        #     columns=["Overall (walking)", "Overall (cycling)"],
        #     scale=True,
        # ),
        #
        # Reduced group — scale and reduce to fewer dimensions:
        # "osm_amenities": GroupConfig(
        #     columns=[...],  # list your columns
        #     scale=True,
        #     reduction_method=ReductionMethod.PCA,
        #     n_components=10,
        # ),
        #
        # Available reduction methods:
        #   ReductionMethod.PCA  — principal component analysis
        #   ReductionMethod.NMF  — non-negative matrix factorisation (requires non-negative input)
        #   ReductionMethod.FA   — factor analysis with varimax rotation
        #   ReductionMethod.ICA  — independent component analysis
    },
)


if __name__ == "__main__":
    input_path: Path = (
        paths.input_file
    )  # you can change this if you've added a new input file

    df = create_feature_set(input_path, config)
    print(f"Created {config.output_name}: {df.shape[0]} rows, {df.shape[1]} columns")
