#!/usr/bin/env python3
"""
Slice a GeoTIFF into a web-mercator tile pyramid ({z}/{x}/{y}.tif).

Uses shared memory and multiprocessing for fast tile generation.

Requirements:
    pip install numpy rasterio mercantile

Usage:
    python make_geotiff_tiles.py input.tif -o tiles/
    python make_geotiff_tiles.py input.tif --zoom-min 8 --zoom-max 14 --workers 32
"""

import argparse
import os
import time
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
import mercantile
from multiprocessing import Pool, shared_memory

WEB_MERCATOR = CRS.from_epsg(3857)
TILE_SIZE = 256


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tile a GeoTIFF into a web-mercator {z}/{x}/{y}.tif (or .png) pyramid",
    )
    parser.add_argument("input", help="Input GeoTIFF file")
    parser.add_argument(
        "-o", "--output-dir", default="tiles",
        help="Output directory for tile pyramid (default: tiles/)",
    )
    parser.add_argument(
        "--zoom-min", type=int, default=8,
        help="Minimum zoom level (default: 8)",
    )
    parser.add_argument(
        "--zoom-max", type=int, default=14,
        help="Maximum zoom level (default: 14)",
    )
    parser.add_argument(
        "--workers", type=int, default=64,
        help="Number of parallel workers (default: 64)",
    )
    parser.add_argument(
        "--png", action="store_true",
        help="Output PNG tiles instead of GeoTIFF (for RGB imagery)",
    )
    return parser.parse_args()


def _init_worker(shm_name, src_shape, src_dtype, src_transform, src_crs, out_dir, use_png):
    """Initializer for pool workers — sets module-level globals."""
    global _SHM_NAME, _SRC_SHAPE, _SRC_DTYPE, _SRC_TRANSFORM, _SRC_CRS, _OUT_DIR, _USE_PNG
    _SHM_NAME = shm_name
    _SRC_SHAPE = src_shape
    _SRC_DTYPE = src_dtype
    _SRC_TRANSFORM = src_transform
    _SRC_CRS = src_crs
    _OUT_DIR = out_dir
    _USE_PNG = use_png


def process_tile(tile):
    shm_r = shared_memory.SharedMemory(name=_SHM_NAME)
    src_arr = np.ndarray(_SRC_SHAPE, dtype=_SRC_DTYPE, buffer=shm_r.buf)

    bounds = mercantile.xy_bounds(tile)
    dst_transform = from_bounds(
        bounds.left, bounds.bottom, bounds.right, bounds.top,
        TILE_SIZE, TILE_SIZE,
    )
    dst_data = np.zeros((_SRC_SHAPE[0], TILE_SIZE, TILE_SIZE), dtype=_SRC_DTYPE)

    reproject(
        source=src_arr,
        destination=dst_data,
        src_transform=_SRC_TRANSFORM,
        src_crs=_SRC_CRS,
        dst_transform=dst_transform,
        dst_crs=WEB_MERCATOR,
        resampling=Resampling.bilinear,
        src_nodata=0,
        dst_nodata=0,
    )

    tile_ext = ".png" if _USE_PNG else ".tif"
    tile_path = os.path.join(_OUT_DIR, str(tile.z), str(tile.x), f"{tile.y}{tile_ext}")

    if _USE_PNG:
        profile = {
            "driver": "PNG",
            "dtype": "uint8",
            "width": TILE_SIZE,
            "height": TILE_SIZE,
            "count": min(_SRC_SHAPE[0], 4),  # RGB or RGBA
        }
        # Clamp to uint8 for PNG
        write_data = dst_data[:min(_SRC_SHAPE[0], 4)]
        if _SRC_DTYPE != np.dtype("uint8"):
            write_data = np.clip(write_data, 0, 255).astype("uint8")
    else:
        profile = {
            "driver": "GTiff",
            "dtype": _SRC_DTYPE,
            "width": TILE_SIZE,
            "height": TILE_SIZE,
            "count": _SRC_SHAPE[0],
            "crs": WEB_MERCATOR,
            "transform": dst_transform,
            "nodata": 0,
            "compress": "deflate",
        }
        write_data = dst_data

    with rasterio.open(tile_path, "w", **profile) as dst:
        dst.write(write_data)

    shm_r.close()
    return tile.z


def main():
    args = parse_args()

    global _SHM_NAME, _SRC_SHAPE, _SRC_DTYPE, _SRC_TRANSFORM, _SRC_CRS, _OUT_DIR, _USE_PNG
    _OUT_DIR = args.output_dir
    _USE_PNG = args.png

    print(f"Reading {args.input}…")
    with rasterio.open(args.input) as src:
        src_data = src.read()
        _SRC_TRANSFORM = src.transform
        _SRC_CRS = src.crs
        src_bounds = src.bounds

    _SRC_SHAPE = src_data.shape
    _SRC_DTYPE = src_data.dtype

    shm = shared_memory.SharedMemory(create=True, size=src_data.nbytes)
    shm_arr = np.ndarray(_SRC_SHAPE, dtype=_SRC_DTYPE, buffer=shm.buf)
    np.copyto(shm_arr, src_data)
    _SHM_NAME = shm.name
    del src_data

    # Build tile list
    all_tiles = []
    for z in range(args.zoom_min, args.zoom_max + 1):
        tiles = list(mercantile.tiles(
            src_bounds.left, src_bounds.bottom,
            src_bounds.right, src_bounds.top, zooms=z,
        ))
        all_tiles.extend(tiles)
        print(f"  Zoom {z}: {len(tiles)} tiles")
    print(f"  Total: {len(all_tiles)} tiles\n")

    for tile in all_tiles:
        d = os.path.join(_OUT_DIR, str(tile.z), str(tile.x))
        os.makedirs(d, exist_ok=True)

    t0 = time.time()
    with Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(_SHM_NAME, _SRC_SHAPE, _SRC_DTYPE, _SRC_TRANSFORM, _SRC_CRS, _OUT_DIR, _USE_PNG),
    ) as pool:
        for i, z in enumerate(pool.imap_unordered(process_tile, all_tiles, chunksize=32), 1):
            if i % 500 == 0 or i == len(all_tiles):
                elapsed = time.time() - t0
                rate = i / elapsed
                print(f"  {i}/{len(all_tiles)} tiles  ({rate:.0f} tiles/s)", flush=True)

    shm.close()
    shm.unlink()

    elapsed = time.time() - t0
    print(f"\nDone! {len(all_tiles)} GeoTIFF tiles in {elapsed:.1f}s ({len(all_tiles)/elapsed:.0f} tiles/s)")


if __name__ == "__main__":
    main()
