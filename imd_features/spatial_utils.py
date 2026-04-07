import os

from project_paths import paths
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape

import json
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# NOTE: This URL is specifically for Bristol, if dataset is extended for other areas, methods for fetching geojson shapes will need to be updated accordingly
URL = "https://opendata.westofengland-ca.gov.uk/api/explore/v2.1/catalog/datasets/lep_lsoa_geog/exports/csv?lang=en&qv1=(bristol)&timezone=Europe%2FLondon&use_labels=true&delimiter=%2C"


def fetch_gdf_and_centroids():
    """Fetch geojson shapes for LSOAs"""

    lsoa_shapes = pd.read_csv(URL)

    # convert geojson string to geometry
    lsoa_shapes["geometry"] = lsoa_shapes["Geo Shape"].apply(
        lambda x: shape(json.loads(x))
    )

    lsoa_shapes = lsoa_shapes[["geometry", "LSOA Code"]].rename(columns={
        "LSOA Code": "lsoa_code",
    })
    
    gdf = gpd.GeoDataFrame(lsoa_shapes, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=27700)
    centroids = np.array(list(zip(gdf.geometry.centroid.x, gdf.geometry.centroid.y)))

    return gdf, centroids

def produce_adjacency_weights(centroids):
    """Produce spatial weights matrix based on adjacency"""
    distances = pairwise_distances(centroids)

    threshold = 0.01  # roughly 700 meters with 1 ~ 111km
    w = (distances < threshold).astype(float)
    w = w / w.sum(axis=1, keepdims=True)
    return w

def produce_cluster_groups(gdf, centroids, n_clusters=10):
    """Produce spatial groups based on clustering of centroids"""

    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(centroids)
    gdf["cluster_id"] = kmeans.labels_

    groups = gdf["cluster_id"].values  
    return groups


def fetch_spatial_support_data():


    if os.path.exists(paths.spatial_weights) and os.path.exists(paths.cluster_groups):
        W = np.load(paths.spatial_weights)
        groups = np.load(paths.cluster_groups)
    else:
        gdf, centroids = fetch_gdf_and_centroids()
        W = produce_adjacency_weights(centroids)
        groups = produce_cluster_groups(gdf, centroids)
        np.save(paths.spatial_weights, W)
        np.save(paths.cluster_groups, groups)

    return W, groups

if __name__ == "__main__":
    W, groups = fetch_spatial_support_data()
    print("Spatial weights matrix shape:", W.shape)
    print("Cluster groups shape:", groups.shape)