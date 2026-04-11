from project_paths import paths, project_root
import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.cluster import KMeans

from pathlib import Path


def load_boundaries(geopackage_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(geopackage_path)
    assert gdf.crs.to_epsg() == 27700, (
        f"Expected EPSG:27700, got {gdf.crs}"
    )  # expected format of geopackage file
    return gdf[["lsoa_code", "geometry"]]


def load_lsoa_to_lad(lookup_path: Path) -> pd.Series:
    lookup = pd.read_csv(lookup_path, usecols=["lsoa_code_21", "lad_name"])
    lookup = lookup.drop_duplicates(subset="lsoa_code_21")
    return lookup.set_index("lsoa_code_21")["lad_name"]


def build_queen_contiguity(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """builds a binary queen contiguity matrix from polygon boundaries.

    Two LSOAs are neighbours if their geometries touch or overlap.
    The diagonal is always zero (an lsoa is not its own neighbour).
    """
    size = len(gdf)
    matrix = np.zeros((size, size), dtype=np.float64)

    pairs = gpd.sjoin(
        gdf, gdf, predicate="touches", how="inner"
    )  # might need to make the predicate touches OR interects?

    idx_left = pairs.index.values
    idx_right = pairs["index_right"].values

    matrix[idx_left, idx_right] = 1.0
    np.fill_diagonal(matrix, 0.0)

    return matrix


def build_spatial_weights(
    gdf: gpd.GeoDataFrame,
    lsoa_codes: list[str],
    lsoa_to_lad: pd.Series,
) -> tuple[np.ndarray, gpd.GeoDataFrame]:
    """Build an adjacency matrix.

    Contiguity is computed within each city independently,
    so lsoas in different cities are never neighbours

    Args:
        gdf: Full boundaries GeoDataFrame from load_boundaries()
        lsoa_codes: from the feature parquet (defines row order)
        lsoa_to_lad: Series mapping lsoa_code to lad_name

    Returns:
        matrix:
        aligned_gdf: GeoDataFrame in the same row order as the matrix, with lad_name attached.
    """

    target = pd.DataFrame(
        {"lsoa_code": lsoa_codes, "row_order": range(len(lsoa_codes))}
    )
    aligned = gdf.merge(target, on="lsoa_code", how="inner").sort_values("row_order")

    matched_codes = set(aligned["lsoa_code"])
    missing = set(lsoa_codes) - matched_codes
    if missing:
        raise ValueError(
            f"{len(missing)} LSOA codes from feature data not found in boundaries: {list(missing)[:5]}"
        )

    aligned = aligned.reset_index(drop=True)
    aligned["lad_name"] = aligned["lsoa_code"].map(lsoa_to_lad)

    unmapped = aligned["lad_name"].isna().sum()
    if unmapped > 0:
        raise ValueError(f"{unmapped} LSOAs have no LAD mapping in the lookup")

    size = len(aligned)
    matrix = np.zeros((size, size), dtype=np.float64)

    for lad_name, city_group in aligned.groupby("lad_name"):
        city_gdf = city_group.reset_index(drop=False).rename(
            columns={"index": "global_idx"}
        )
        city_gdf_for_sjoin = city_gdf.set_index(city_gdf.index)[["geometry"]].copy()

        pairs = gpd.sjoin(
            city_gdf_for_sjoin, city_gdf_for_sjoin, predicate="touches", how="inner"
        )

        local_left = pairs.index.values
        local_right = pairs["index_right"].values

        global_left = city_gdf.iloc[local_left]["global_idx"].values
        global_right = city_gdf.iloc[local_right]["global_idx"].values

        matrix[global_left, global_right] = 1.0

    np.fill_diagonal(matrix, 0.0)

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    matrix = matrix / row_sums

    return matrix, aligned


def produce_cluster_groups(gdf, n_clusters_per_city=10):
    """Produce spatial groups based on clustering of centroids"""

    groups = np.empty(len(gdf), dtype=np.int32)
    cluster_offset = 0

    for lad_name, city_group in gdf.groupby("lad_name"):
        centroids = np.column_stack(
            [
                city_group.geometry.centroid.x,
                city_group.geometry.centroid.y,
            ]
        )

        k = min(n_clusters_per_city, len(city_group))
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(centroids)

        groups[city_group.index] = labels + cluster_offset
        cluster_offset += k

    return groups


def fetch_spatial_support_data(
    lsoa_codes: list[str],
    boundaries_path: Path,
    lookup_path: Path,
    n_clusters_per_city: int = 10,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """top level util to build matrix (W), groups, and aligned LSOA codes.

    Args:
        lsoa_codes: LSOA codes from the feature parquet (defines row order).
        boundaries_path: Path to ONS BGC V5 GeoPackage.
        lookup_path: Path to ONS lsoa_lookup.csv.
        n_clusters_per_city: KMeans clusters per city for GroupKFold.

    Returns:
        W: matrix, shape (n, n).
        groups: Cluster group ids for GroupKFold, shape (n,).
        aligned_lsoa_codes: lsoa codes in the row order of W and groups.
    """
    gdf = load_boundaries(boundaries_path)
    lsoa_to_lad = load_lsoa_to_lad(lookup_path)

    W, aligned_gdf = build_spatial_weights(gdf, lsoa_codes, lsoa_to_lad)
    groups = produce_cluster_groups(aligned_gdf, n_clusters_per_city)

    aligned_lsoa_codes = aligned_gdf["lsoa_code"].tolist()

    return W, groups, aligned_lsoa_codes


if __name__ == "__main__":
    import polars as pl

    df = pl.read_parquet(
        project_root / "data" / "input" / "combined_data_multi_city.parquet"
    )
    lsoa_codes = df.get_column("lsoa_code").to_list()

    W, groups, aligned_codes = fetch_spatial_support_data(
        lsoa_codes=lsoa_codes,
        boundaries_path=paths.polygons,
        lookup_path=paths.lads,
    )

    np.fill_diagonal(W, 0)
    print(f"Shape: {W.shape}")
    print(f"Non-zero off-diagonal entries: {np.count_nonzero(W)}")
    print(f"Mean neighbours per LSOA: {(W > 0).sum(axis=1).mean():.1f}")
    print(f"Any LSOA with 0 neighbours: {(W.sum(axis=1) == 0).any()}")
    print(f"Unique cluster groups: {len(np.unique(groups))}")
    print(f"Aligned codes match input: {aligned_codes == lsoa_codes}")
