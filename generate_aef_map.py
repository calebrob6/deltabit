#!/usr/bin/env python3                                                                                                                                                                                  [112/209]
"""
Generate an AEF embedding diff (2024 - 2020) PCA3 RGB GeoTIFF
over the bounds of a Sentinel-2 T10TET tile (Seattle area).

Requirements:
    pip install numpy xarray zarr rasterix scikit-learn rasterio affine pyproj s3fs

Usage:
    python generate_aef_diff.py [--output aef_diff_pca3_seattle.tif]
"""

import argparse
import time
import warnings
import numpy as np
import xarray as xr
import zarr
import zarr.storage
from zarr.errors import ZarrUserWarning
import rasterio
from rasterio.transform import from_bounds
from sklearn.decomposition import PCA
from rasterix import RasterIndex
from affine import Affine
from pyproj import CRS

warnings.filterwarnings("ignore", category=ZarrUserWarning)

STORE_URL = "s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic"

# Bounding box of S2 tile T10TET in EPSG:4326
# (derived from the 10980x10980 UTM10N tile)
MINX, MINY, MAXX, MAXY = (
    -123.00026735765742,
    46.856639881437616,
    -121.53272262323941,
    47.85370184201499,
)


def open_aef_mosaic():
    print("Opening AEF mosaic...")
    tic = time.time()
    store = zarr.storage.FsspecStore.from_url(
        STORE_URL,
        storage_options={"anon": True},
        read_only=True,
    )
    ds = xr.open_zarr(store, zarr_format=3, consolidated=False)
    print(f"  Opened in {time.time() - tic:.2f}s")
    return ds


def assign_rasterix_index(ds):
    attrs = ds.attrs
    affine = Affine(*attrs["spatial:transform"])
    crs = CRS.from_user_input(attrs["proj:code"])
    height, width = attrs["spatial:shape"]

    raster_idx = RasterIndex.from_transform(
        affine, width=width, height=height,
        x_dim="x", y_dim="y", crs=crs,
    )

    emb = ds["embeddings"].drop_vars(["x", "y"], errors="ignore")
    emb = emb.assign_coords(raster_idx.create_variables())
    return emb


def main():
    parser = argparse.ArgumentParser(description="Generate AEF diff PCA3 GeoTIFF")
    parser.add_argument(
        "--output", "-o",
        default="aef_diff_pca3_seattle.tif",
        help="Output GeoTIFF path (default: aef_diff_pca3_seattle.tif)",
    )
    args = parser.parse_args()

    ds = open_aef_mosaic()
    embedding_ds = assign_rasterix_index(ds)

    # Subset for 2024 and 2020
    print("Subsetting 2024 embeddings...")
    subset_2024 = embedding_ds.sel(
        x=slice(MINX, MAXX),
        y=slice(MAXY, MINY),
        time=2024,
    )
    print(f"  Shape: {subset_2024.shape}")

    print("Subsetting 2020 embeddings...")
    subset_2020 = embedding_ds.sel(
        x=slice(MINX, MAXX),
        y=slice(MAXY, MINY),
        time=2020,
    )
    print(f"  Shape: {subset_2020.shape}")

    # Download data
    print("Computing 2024 data...")
    tic = time.time()
    data_2024 = subset_2024.compute().values  # native int8
    print(f"  Done in {time.time() - tic:.2f}s, shape: {data_2024.shape}")

    print("Computing 2020 data...")
    tic = time.time()
    data_2020 = subset_2020.compute().values  # native int8
    print(f"  Done in {time.time() - tic:.2f}s, shape: {data_2020.shape}")

    # Save intermediate AEF rasters (int8, matching source)
    num_channels, height, width = data_2024.shape
    y_coords = subset_2024.y.values
    x_coords = subset_2024.x.values
    intermediate_transform = from_bounds(
        x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max(),
        width, height,
    )
    intermediate_profile = {
        "driver": "GTiff",
        "dtype": "int8",
        "width": width,
        "height": height,
        "count": num_channels,
        "crs": "EPSG:4326",
        "transform": intermediate_transform,
        "compress": "zstd",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "bigtiff": "yes",
    }

    aef_2024_path = args.output.replace(".tif", "_aef_2024.tif")
    print(f"Saving 2024 AEF raster to {aef_2024_path}...")
    with rasterio.open(aef_2024_path, "w", **intermediate_profile) as dst:
        dst.write(data_2024)

    aef_2020_path = args.output.replace(".tif", "_aef_2020.tif")
    print(f"Saving 2020 AEF raster to {aef_2020_path}...")
    with rasterio.open(aef_2020_path, "w", **intermediate_profile) as dst:
        dst.write(data_2020)

    # Diff (cast to float32 to avoid int8 overflow)
    print("Computing diff (2024 - 2020)...")
    diff = data_2024.astype(np.float32) - data_2020.astype(np.float32)
    del data_2024, data_2020
    print(f"  Diff shape: {diff.shape}")

    # PCA
    print("Running PCA (3 components)...")
    n_components = 3
    pca = PCA(n_components=n_components)
    reshaped = diff.reshape(num_channels, -1).T
    pca.fit(reshaped[::1000])
    pca_transformed = pca.transform(reshaped)
    pca_transformed = pca_transformed.reshape(height, width, n_components)

    # Percentile normalize to [0, 1]
    print("Percentile normalizing...")
    pca_transformed = pca_transformed - np.percentile(pca_transformed, 1, axis=(0, 1))
    scale = np.percentile(pca_transformed, 99, axis=(0, 1))
    scale[scale == 0] = 1.0
    pca_transformed = pca_transformed / scale
    pca_transformed = np.clip(pca_transformed, 0, 1)

    # Save as RGB GeoTIFF
    print(f"Saving to {args.output}...")
    rgb = (pca_transformed * 255).astype(np.uint8)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": width,
        "height": height,
        "count": 3,
        "crs": "EPSG:4326",
        "transform": intermediate_transform,
        "compress": "deflate",
    }

    with rasterio.open(args.output, "w", **profile) as dst:
        for i in range(3):
            dst.write(rgb[:, :, i], i + 1)

    print(f"Done! Saved {args.output}")

if __name__ == "__main__":
    main()