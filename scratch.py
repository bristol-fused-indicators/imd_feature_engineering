import polars as pl
from project_paths import paths
from pprint import pprint

df = pl.read_parquet(paths.input_file)

# pprint(df.columns)


df.fill_nan(0).fill_null(0).write_parquet(paths.input_file)
