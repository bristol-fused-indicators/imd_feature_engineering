# IMD Feature Engineering

Feature selection and dimensionality reduction for the Bristol IMD nowcasting project. Takes the combined LSOA grain dataset from the [IMD data pipeline](https://github.com/bristol-fused-indicators/imd-dataset-pipeline) and produces named, versioned feature sets for modelling experiments.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
```bash
git clone <repo-url>
cd imd-feature-engineering
uv sync
```

Place the combined indicators parquet from the pipeline repo into `data/input/`.

## Quick start

Copy the template and edit it:
```bash
cp configs/template.py configs/my_experiment.py
```

Define your feature groups, then run:
```bash
uv run python configs/my_experiment.py
```

Output lands in `data/output/` as a parquet file with a config hson manifest file.

OR

copy the notebook `notbooks/0_demo_create_and_inspect.ipynb` and change the config in there, then run the notebook.

## How it works

A feature set config defines one or more **groups** of columns from the input data. Each group can optionally be scaled (StandardScaler) and reduced (PCA, NMF, Factor Analysis, ICA). Groups cannot share columns.

```python
config = FeatureSetConfig(
    name="my_experiment",
    description="What this config is testing",
    groups={
        "crime": GroupConfig(
            columns=["violent-crime", "burglary", "drugs"],
            scale=True,
        ),
        "osm_amenities": GroupConfig(
            columns=["count_healthcare_access_500", "count_healthcare_access_1000"],
            scale=True,
            reduction_method=ReductionMethod.PCA,
            n_components=5,
        ),
    },
)
```

Output files are named `{name}_{config_hash}.parquet`. The hash is derived from the config, so identical configs always produce the same filename.

## Inspecting results

`imd_features/inspect.py` provides diagnostic functions for examining feature sets:

- `group_summary(config, metadata)` - table of group sizes, reduction methods, and diagnostics
- `correlation_within_groups(df, config)` - heatmap per group
- `correlation_between_groups(df, config)` - mean absolute correlation between groups
- `correlation_full(df)` - full feature correlation matrix
- `distribution_plot(df, config)` - box plots per group

See `notebooks/` for worked examples.

## Recreating feature sets

Config manifest jsonss are committed alongside config scripts. To recreate all feature sets from existing configs:
```bash
uv run python configs/recreate.py
```


## Project structure
```
configs/            Config scripts — one per experiment, committed
data/
  input/            Combined parquet from pipeline (gitignored)
  reference/        IMD target data (committed)
  output/           Produced feature sets + config sidecars (gitignored)
imd_features/       Core package
  config.py         FeatureSetConfig, GroupConfig, ReductionMethod
  process.py        Feature set production
  inspect.py        Diagnostics and visualisation
  evaluate.py       Screening evaluation (TODO)
notebooks/          Shared notebooks for inspection and comparison
tests/              Smoke tests
```