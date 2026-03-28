import os, sys, time
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
import mercantile
from multiprocessing import Pool, shared_memory

SRC = 'aef_diff_pca8_2024_minus_2020_uint8.tif'
OUT_DIR = 'tiles'
TILE_SIZE = 256
ZOOM_MIN, ZOOM_MAX = 8, 14
WEB_MERCATOR = CRS.from_epsg(3857)
NUM_WORKERS = 64

# Read source raster into shared memory so workers don't each read the file
with rasterio.open(SRC) as src:
    src_data = src.read()
    src_profile = dict(src.profile)
    src_transform = src.transform
    src_crs = src.crs
    src_bounds = src.bounds
    src_count = src.count

shm = shared_memory.SharedMemory(create=True, size=src_data.nbytes)
shm_arr = np.ndarray(src_data.shape, dtype=src_data.dtype, buffer=shm.buf)
np.copyto(shm_arr, src_data)
SHM_NAME = shm.name
SRC_SHAPE = src_data.shape
SRC_DTYPE = src_data.dtype
del src_data

# Build tile list
all_tiles = []
for z in range(ZOOM_MIN, ZOOM_MAX + 1):
    tiles = list(mercantile.tiles(
        src_bounds.left, src_bounds.bottom,
        src_bounds.right, src_bounds.top, zooms=z
    ))
    all_tiles.extend(tiles)
    print(f"  Zoom {z}: {len(tiles)} tiles")
print(f"  Total: {len(all_tiles)} tiles\n")

# Pre-create directories
for tile in all_tiles:
    d = os.path.join(OUT_DIR, str(tile.z), str(tile.x))
    os.makedirs(d, exist_ok=True)

def process_tile(tile):
    shm_r = shared_memory.SharedMemory(name=SHM_NAME)
    src_arr = np.ndarray(SRC_SHAPE, dtype=SRC_DTYPE, buffer=shm_r.buf)

    bounds = mercantile.xy_bounds(tile)
    dst_transform = from_bounds(
        bounds.left, bounds.bottom, bounds.right, bounds.top,
        TILE_SIZE, TILE_SIZE
    )
    dst_data = np.zeros((SRC_SHAPE[0], TILE_SIZE, TILE_SIZE), dtype=np.uint8)

    reproject(
        source=src_arr,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=WEB_MERCATOR,
        resampling=Resampling.bilinear,
        src_nodata=0,
        dst_nodata=0,
    )

    tile_path = os.path.join(OUT_DIR, str(tile.z), str(tile.x), f"{tile.y}.tif")
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': TILE_SIZE,
        'height': TILE_SIZE,
        'count': SRC_SHAPE[0],
        'crs': WEB_MERCATOR,
        'transform': dst_transform,
        'nodata': 0,
        'compress': 'deflate',
    }
    with rasterio.open(tile_path, 'w', **profile) as dst:
        dst.write(dst_data)

    shm_r.close()
    return tile.z

t0 = time.time()
with Pool(NUM_WORKERS) as pool:
    done = {z: 0 for z in range(ZOOM_MIN, ZOOM_MAX + 1)}
    for i, z in enumerate(pool.imap_unordered(process_tile, all_tiles, chunksize=32), 1):
        done[z] += 1
        if i % 500 == 0 or i == len(all_tiles):
            elapsed = time.time() - t0
            rate = i / elapsed
            print(f"  {i}/{len(all_tiles)} tiles  ({rate:.0f} tiles/s)", flush=True)

shm.close()
shm.unlink()

elapsed = time.time() - t0
print(f"\nDone! {len(all_tiles)} GeoTIFF tiles in {elapsed:.1f}s ({len(all_tiles)/elapsed:.0f} tiles/s)")
