"""Utilities for converting Sentinel-2 MGRS tile IDs to EPSG:4326 bounding boxes."""

import mgrs as _mgrs
from pyproj import Transformer

# Sentinel-2 tiles are 10980 × 10980 pixels at 10 m → 109,800 m per side.
S2_TILE_SIZE_M = 109_800


def parse_mgrs_tile_id(tile_id):
    """Normalize a Sentinel-2 MGRS tile ID.

    Accepts ``'10TET'`` or ``'T10TET'`` (the ``T`` prefix is common in S2
    naming).  Returns the 5-character canonical form (e.g. ``'10TET'``).
    """
    tile_id = tile_id.strip().upper()
    if len(tile_id) == 6 and tile_id[0] == "T" and tile_id[1:3].isdigit():
        tile_id = tile_id[1:]
    if len(tile_id) != 5 or not tile_id[:2].isdigit():
        raise ValueError(
            f"Invalid MGRS tile ID '{tile_id}'. "
            "Expected 5 characters like '10TET' or 6 characters like 'T10TET'."
        )
    return tile_id


def mgrs_tile_bounds(tile_id):
    """Return the EPSG:4326 bounding box for a Sentinel-2 MGRS tile.

    The Sentinel-2 tiling grid places the image origin at the **NW corner** of
    the MGRS 100 km grid square.  The tile extends 109,800 m (10,980 px × 10 m)
    south and east from that origin.

    Parameters
    ----------
    tile_id : str
        MGRS tile identifier, e.g. ``'10TET'`` or ``'T10TET'``.

    Returns
    -------
    tuple of (minx, miny, maxx, maxy)
        Axis-aligned bounding box in EPSG:4326 (longitude / latitude).
    """
    tile_id = parse_mgrs_tile_id(tile_id)

    # --- Determine the UTM origin of the 100 km grid square ----------------
    m = _mgrs.MGRS()
    # Easting/northing = 0 within the 100 km square → SW corner of the square.
    sw_lat, sw_lon = m.toLatLon(f"{tile_id}0000000000")  # 1 m precision

    zone_num = int(tile_id[:2])
    band_letter = tile_id[2]
    epsg = (32600 + zone_num) if band_letter >= "N" else (32700 + zone_num)

    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_wgs = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    origin_e, origin_n = to_utm.transform(sw_lon, sw_lat)
    # origin_e, origin_n is at the SW corner of the 100 km square.

    # --- S2 tile extent in UTM ----------------------------------------------
    # NW corner of the 100 km square = (origin_e, origin_n + 100 000).
    # The tile extends 109 800 m south and east from that NW corner.
    tile_emin = origin_e
    tile_nmax = origin_n + 100_000
    tile_emax = tile_emin + S2_TILE_SIZE_M
    tile_nmin = tile_nmax - S2_TILE_SIZE_M

    # --- Convert UTM rectangle → WGS-84 envelope ---------------------------
    # Sample all four edges (not just corners) because UTM edges curve in
    # geographic coordinates.  500 points per edge is more than enough.
    n_samples = 500
    lons, lats = [], []
    for i in range(n_samples + 1):
        t = i / n_samples
        e = tile_emin + t * (tile_emax - tile_emin)
        n = tile_nmin + t * (tile_nmax - tile_nmin)
        for ex, ny in [
            (e, tile_nmin),   # south edge
            (e, tile_nmax),   # north edge
            (tile_emin, n),   # west edge
            (tile_emax, n),   # east edge
        ]:
            lon, lat = to_wgs.transform(ex, ny)
            lons.append(lon)
            lats.append(lat)

    return min(lons), min(lats), max(lons), max(lats)


def mgrs_tile_center(tile_id):
    """Return the approximate center of a Sentinel-2 MGRS tile in (lat, lon)."""
    minx, miny, maxx, maxy = mgrs_tile_bounds(tile_id)
    return (miny + maxy) / 2, (minx + maxx) / 2


def mgrs_tile_utm_epsg(tile_id):
    """Return the EPSG code of the UTM zone for a given MGRS tile."""
    tile_id = parse_mgrs_tile_id(tile_id)
    zone_num = int(tile_id[:2])
    band_letter = tile_id[2]
    return (32600 + zone_num) if band_letter >= "N" else (32700 + zone_num)
