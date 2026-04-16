#!/usr/bin/env python3
"""
Download a low-cloud Sentinel-2 true-color scene for an MGRS tile and year.

Queries the Element 84 Earth Search STAC catalog for the least-cloudy
Sentinel-2 L2A scene covering the requested tile in the given year, then
downloads the TCI (true-color image) and saves it as a GeoTIFF.

The output can be tiled with ``make_geotiff_tiles.py --png`` for use in the
browser visualizer.

Requirements:
    pip install rasterio pystac-client mgrs pyproj

Usage:
    python download_s2_scene.py --tile T10TET --year 2020 2025
    python download_s2_scene.py --tile 33UUP --year 2022 --max-cloud 5
"""

import argparse
import os
import time

import rasterio
from rasterio.transform import from_bounds
from pystac_client import Client

from mgrs_utils import parse_mgrs_tile_id, mgrs_tile_bounds

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"


def find_best_scene(tile_id, year, max_cloud=10, max_nodata=5):
    """Find the least-cloudy S2 scene for *tile_id* in *year*.

    Returns
    -------
    pystac.Item or None
    """
    print(f"Searching for S2 scenes: tile={tile_id}, year={year}, "
          f"cloud<{max_cloud}%, nodata<{max_nodata}%...")

    catalog = Client.open(STAC_URL)

    # Filter by MGRS tile using the grid:code property
    search = catalog.search(
        collections=[COLLECTION],
        datetime=f"{year}-01-01/{year}-12-31",
        query={
            "grid:code": {"eq": f"MGRS-{tile_id}"},
            "eo:cloud_cover": {"lt": max_cloud},
            "s2:nodata_pixel_percentage": {"lt": max_nodata},
        },
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        max_items=10,
    )

    items = list(search.items())
    if not items:
        print(f"  No scenes found with cloud<{max_cloud}% and nodata<{max_nodata}%")
        return None

    # Pick the one with lowest cloud cover
    best = min(items, key=lambda it: it.properties.get("eo:cloud_cover", 100))
    cc = best.properties.get("eo:cloud_cover", "?")
    nd = best.properties.get("s2:nodata_pixel_percentage", "?")
    print(f"  Best scene: {best.id}")
    print(f"    Date:    {best.datetime.strftime('%Y-%m-%d')}")
    print(f"    Cloud:   {cc}%")
    print(f"    Nodata:  {nd}%")
    return best


def download_tci(item, output_path):
    """Download the TCI (true-color) asset from a STAC item."""
    # Try 'visual' first, then 'tci'
    asset_key = None
    for key in ("visual", "tci", "true-color"):
        if key in item.assets:
            asset_key = key
            break

    if asset_key is None:
        raise RuntimeError(
            f"No TCI/visual asset found in item {item.id}. "
            f"Available assets: {list(item.assets.keys())}"
        )

    href = item.assets[asset_key].href
    print(f"  Downloading {asset_key}: {href}")

    tic = time.time()

    # GDAL environment for reading COGs from S3/HTTP
    env_opts = {
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
    }

    with rasterio.Env(**env_opts):
        with rasterio.open(href) as src:
            data = src.read()
            profile = src.profile.copy()

    elapsed = time.time() - tic
    nbands, height, width = data.shape
    print(f"  Downloaded in {elapsed:.1f}s — {nbands} bands, {width}×{height}, "
          f"dtype={data.dtype}")

    # Write locally with compression
    profile.update(
        driver="GTiff",
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    # Remove JPEG-specific options that may be inherited from the COG
    for key in ("jpeg_quality", "quality"):
        profile.pop(key, None)

    print(f"  Writing {output_path}...")
    tic = time.time()
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Saved ({size_mb:.1f} MB) in {time.time() - tic:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Download non-cloudy Sentinel-2 scenes for a given MGRS tile and year(s)",
    )
    parser.add_argument(
        "--tile", "-t",
        required=True,
        help="Sentinel-2 MGRS tile ID (e.g. T10TET, 33UUP)",
    )
    parser.add_argument(
        "--year", "-y",
        type=int,
        nargs="+",
        required=True,
        help="Year(s) to download scenes for (e.g. 2020 2025)",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=10,
        help="Maximum cloud cover percentage (default: 10)",
    )
    parser.add_argument(
        "--max-nodata",
        type=float,
        default=5,
        help="Maximum nodata pixel percentage (default: 5)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Output directory for downloaded scenes (default: current directory)",
    )
    args = parser.parse_args()

    tile_id = parse_mgrs_tile_id(args.tile)

    if args.output_dir != ".":
        os.makedirs(args.output_dir, exist_ok=True)

    for year in args.year:
        item = find_best_scene(
            tile_id, year,
            max_cloud=args.max_cloud,
            max_nodata=args.max_nodata,
        )
        if item is None:
            print(f"  Skipping year {year} — no suitable scene found.\n"
                  f"  Try increasing --max-cloud or --max-nodata.\n")
            continue

        date_str = item.datetime.strftime("%Y%m%d")
        output_path = os.path.join(
            args.output_dir, f"s2_{tile_id}_{date_str}.tif"
        )
        download_tci(item, output_path)
        print()

    print("Done!")


if __name__ == "__main__":
    main()
