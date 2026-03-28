#!/usr/bin/env python3
"""
Download AEF embeddings for a Sentinel-2 MGRS tile and save as GeoTIFFs.

The output is a multi-band (64-band) int8 GeoTIFF in EPSG:4326.

Requirements:
    pip install numpy xarray zarr rasterio affine pyproj s3fs rasterix mgrs

Usage:
    python download_aef_geotiff.py --tile T10TET
    python download_aef_geotiff.py --tile T10TET --year 2020 2025
    python download_aef_geotiff.py --tile 33UUP --year 2020 --output custom.tif
"""

import argparse
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import xarray as xr
import zarr
import zarr.storage
from zarr.errors import ZarrUserWarning
import rasterio
from rasterio.transform import from_bounds
from rasterix import RasterIndex
from affine import Affine
from pyproj import CRS

from mgrs_utils import mgrs_tile_bounds, parse_mgrs_tile_id

warnings.filterwarnings("ignore", category=ZarrUserWarning)

STORE_URL = "s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic"
VALID_YEARS = list(range(2017, 2026))


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


def download_year(embedding_ds, year, bounds):
    minx, miny, maxx, maxy = bounds
    print(f"Subsetting {year} embeddings...")
    subset = embedding_ds.sel(
        x=slice(minx, maxx),
        y=slice(maxy, miny),
        time=year,
    )
    print(f"  Shape: {subset.shape}")

    print(f"Downloading {year} data...")
    tic = time.time()
    data = subset.compute()
    elapsed = time.time() - tic
    print(f"  Done in {elapsed:.1f}s, shape: {data.shape}, dtype: {data.dtype}")
    return data


def save_geotiff(data, output_path):
    values = data.values  # keep native int8
    num_channels, height, width = values.shape

    y_coords = data.y.values
    x_coords = data.x.values
    transform = from_bounds(
        x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max(),
        width, height,
    )

    profile = {
        "driver": "GTiff",
        "dtype": values.dtype.name,
        "width": width,
        "height": height,
        "count": num_channels,
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "zstd",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "bigtiff": "yes",
    }

    print(f"Writing {output_path} ({num_channels} bands, {width}x{height}, {values.dtype})...")
    tic = time.time()
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(values)
    elapsed = time.time() - tic

    import os
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Saved ({size_mb:.1f} MB) in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Download AEF embeddings for a Sentinel-2 MGRS tile and save as GeoTIFF",
    )
    parser.add_argument(
        "--tile", "-t",
        default="T10TET",
        help="Sentinel-2 MGRS tile ID (e.g. T10TET, 33UUP). Default: T10TET (Seattle)",
    )
    parser.add_argument(
        "--year", "-y",
        type=int,
        nargs="+",
        default=[2020, 2025],
        help=f"Year(s) to download (valid: {VALID_YEARS[0]}-{VALID_YEARS[-1]}, default: 2020 2025)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path template (default: aef_{tile}_{year}.tif)",
    )
    args = parser.parse_args()

    tile_id = parse_mgrs_tile_id(args.tile)

    for year in args.year:
        if year not in VALID_YEARS:
            parser.error(f"Invalid year {year}. Must be in {VALID_YEARS[0]}-{VALID_YEARS[-1]}")

    print(f"Computing bounding box for MGRS tile {tile_id}...")
    bounds = mgrs_tile_bounds(tile_id)
    print(f"  Bounds (EPSG:4326): {bounds}")

    ds = open_aef_mosaic()
    embedding_ds = assign_rasterix_index(ds)

    def _download_and_save(year):
        data = download_year(embedding_ds, year, bounds)
        if args.output and len(args.year) == 1:
            output_path = args.output
        else:
            output_path = f"aef_{tile_id}_{year}.tif"
        save_geotiff(data, output_path)
        return year, output_path

    with ThreadPoolExecutor(max_workers=len(args.year)) as pool:
        futures = {pool.submit(_download_and_save, y): y for y in args.year}
        for future in as_completed(futures):
            year, path = future.result()
            print(f"  Completed {year} → {path}")

    print("Done!")


if __name__ == "__main__":
    main()
