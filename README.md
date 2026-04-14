# Deltabit

Interactive change-detection tool for satellite imagery. Swipe between two dates
of Sentinel-2 imagery, label pixels as "change" or "no change", train a logistic
regression on [AEF](https://source.coop/tge-labs/aef-mosaic) embeddings, and see
predictions in real time — all in the browser via WebGPU (or CPU fallback).

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/WebGPU-supported-green" alt="WebGPU">
</p>

## Quick Start

```bash
# Set up environment
conda env create -f environment.yml
conda activate in-browser-embedding

# Download AEF embeddings for Seattle (2020 + 2025)
python download_aef_geotiff.py

# Compute embedding diff and reduce to PCA components
python pca_diff.py aef_seattle_2020.tif aef_seattle_2025.tif -n 3

# Generate web-mercator tile pyramid
python make_geotiff_tiles.py aef_diff_pca3_seattle_2025_minus_seattle_2020.tif -o tiles/

# Launch the viewer
python -m http.server 8000
# Open http://localhost:8000/visualizer/interactive.html?tiles=../tiles/{z}/{x}/{y}.tif
```

## How It Works

The pipeline downloads 64-band int8 AEF (AI Earth Foundation) embeddings from a
public Zarr v3 mosaic on S3, computes a per-pixel temporal diff, reduces it via
PCA, and tiles the result for browser consumption.

```
download_aef_geotiff.py     Fetch 64-band embeddings for any year (2017–2025)
         │
         ▼
    pca_diff.py              Diff two years → PCA reduction
         │
         ▼
 make_geotiff_tiles.py       Slice GeoTIFF → {z}/{x}/{y}.tif web-mercator pyramid
         │
         ▼
  visualizer/*.html          Leaflet side-by-side viewer with labeling + in-browser ML
```

`generate_aef_map.py` wraps the first two steps into a single script.

## Scripts

### `download_aef_geotiff.py`

Downloads 64-band AEF embedding GeoTIFFs for the Seattle area (Sentinel-2 tile
T10TET). Multiple years download in parallel.

```bash
python download_aef_geotiff.py                          # defaults: 2020 + 2025
python download_aef_geotiff.py --year 2018 2022         # specific years
python download_aef_geotiff.py -y 2020 -o my_2020.tif   # custom output
```

### `pca_diff.py`

Pixel-wise diff of two AEF GeoTIFFs → PCA reduction. Stores explained variance
in GeoTIFF metadata tags.

```bash
python pca_diff.py aef_seattle_2020.tif aef_seattle_2025.tif           # 8 components
python pca_diff.py aef_seattle_2020.tif aef_seattle_2025.tif -n 3      # 3 components
python pca_diff.py aef_seattle_2020.tif aef_seattle_2025.tif --subsample 500
```

### `make_geotiff_tiles.py`

Slices any GeoTIFF into a web-mercator `{z}/{x}/{y}.tif` tile pyramid using
multiprocessing + shared memory.

```bash
python make_geotiff_tiles.py input.tif                               # zoom 8–14
python make_geotiff_tiles.py input.tif -o out/ --zoom-min 10 --zoom-max 16
python make_geotiff_tiles.py input.tif --workers 8                   # fewer CPUs
```

### `generate_aef_map.py`

All-in-one: downloads two years of AEF data, diffs, and outputs a PCA-3 RGB
GeoTIFF for the Seattle area.

```bash
python generate_aef_map.py --output aef_diff_pca3.tif
```

## Visualizer

The HTML viewers (`visualizer/`) are self-contained — no build step, no
bundler. They use Leaflet + GeoTIFF.js and load embedding tiles as
`{z}/{x}/{y}.tif` pyramids. Override the tile URL with `?tiles=`.

### Keyboard Shortcuts

| Key   | Action                    |
|-------|---------------------------|
| `A`   | Swipe → show earlier date |
| `D`   | Swipe → show later date   |
| `S`   | Swipe → center            |
| `C`   | Label: Change             |
| `N`   | Label: No Change          |
| `Esc` | Stop labeling             |
| `T`   | Toggle binary overlay     |
| `H`   | Toggle heatmap overlay    |
| `G`   | Train model               |

## Data Source

Embeddings come from the [AEF Mosaic](https://source.coop/tge-labs/aef-mosaic)
on Source Cooperative — anonymous S3 access, no credentials needed. Coverage
spans 2017–2025 at 10 m resolution (Sentinel-2 grid).