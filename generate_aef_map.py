#!/usr/bin/env python3
"""
Generate an AEF embedding diff PCA3 RGB GeoTIFF for any Sentinel-2 MGRS tile.

Requirements:
    pip install numpy xarray zarr rasterix scikit-learn rasterio affine pyproj s3fs mgrs

Usage:
    python generate_aef_map.py --tile T10TET
    python generate_aef_map.py --tile 33UUP --output aef_diff_pca3_33UUP.tif
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

from mgrs_utils import mgrs_tile_bounds, parse_mgrs_tile_id

warnings.filterwarnings("ignore", category=ZarrUserWarning)

STORE_URL = "s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic"


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
        "--tile", "-t",
        default="T10TET",
        help="Sentinel-2 MGRS tile ID (e.g. T10TET, 33UUP). Default: T10TET (Seattle)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output GeoTIFF path (default: aef_diff_pca3_{tile}.tif)",
    )
    parser.add_argument(
        "--year-a", type=int, default=2020,
        help="Earlier year (default: 2020)",
    )
    parser.add_argument(
        "--year-b", type=int, default=2024,
        help="Later year (default: 2024)",
    )
    args = parser.parse_args()

    tile_id = parse_mgrs_tile_id(args.tile)
    if args.output is None:
        args.output = f"aef_diff_pca3_{tile_id}.tif"

    print(f"Computing bounding box for MGRS tile {tile_id}...")
    bounds = mgrs_tile_bounds(tile_id)
    minx, miny, maxx, maxy = bounds
    print(f"  Bounds (EPSG:4326): {bounds}")

    ds = open_aef_mosaic()
    embedding_ds = assign_rasterix_index(ds)

    # Subset for year_b and year_a
    print(f"Subsetting {args.year_b} embeddings...")
    subset_b = embedding_ds.sel(
        x=slice(minx, maxx),
        y=slice(maxy, miny),
        time=args.year_b,
    )
    print(f"  Shape: {subset_b.shape}")

    print(f"Subsetting {args.year_a} embeddings...")
    subset_a = embedding_ds.sel(
        x=slice(minx, maxx),
        y=slice(maxy, miny),
        time=args.year_a,
    )
    print(f"  Shape: {subset_a.shape}")

    # Download data
    print(f"Computing {args.year_b} data...")
    tic = time.time()
    data_b = subset_b.compute().values  # native int8
    print(f"  Done in {time.time() - tic:.2f}s, shape: {data_b.shape}")

    print(f"Computing {args.year_a} data...")
    tic = time.time()
    data_a = subset_a.compute().values  # native int8
    print(f"  Done in {time.time() - tic:.2f}s, shape: {data_a.shape}")

    # Save intermediate AEF rasters (int8, matching source)
    num_channels, height, width = data_b.shape
    y_coords = subset_b.y.values
    x_coords = subset_b.x.values
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

    aef_b_path = args.output.replace(".tif", f"_aef_{args.year_b}.tif")
    print(f"Saving {args.year_b} AEF raster to {aef_b_path}...")
    with rasterio.open(aef_b_path, "w", **intermediate_profile) as dst:
        dst.write(data_b)

    aef_a_path = args.output.replace(".tif", f"_aef_{args.year_a}.tif")
    print(f"Saving {args.year_a} AEF raster to {aef_a_path}...")
    with rasterio.open(aef_a_path, "w", **intermediate_profile) as dst:
        dst.write(data_a)

    # Diff (cast to float32 to avoid int8 overflow)
    print(f"Computing diff ({args.year_b} - {args.year_a})...")
    diff = data_b.astype(np.float32) - data_a.astype(np.float32)
    del data_b, data_a
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