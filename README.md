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
```

### Seattle (MGRS tile T10TET) — full walkthrough

```bash
# 1. Download Sentinel-2 true-color scenes (finds least-cloudy per year)
python download_s2_scene.py --tile T10TET --year 2020 2025

# 2. Download AEF embeddings (64-band int8 GeoTIFFs)
python download_aef_geotiff.py --tile T10TET --year 2020 2025

# 3. Compute embedding diff → PCA reduction
python pca_diff.py aef_10TET_2020.tif aef_10TET_2025.tif -n 3

# 4. Generate tile pyramids
#    Embedding diff tiles (GeoTIFF — used by the interactive labeling viewer)
python make_geotiff_tiles.py aef_diff_pca3_10TET_2025_minus_10TET_2020.tif -o tiles/

#    Sentinel-2 imagery tiles (PNG — used for the side-by-side comparison)
python make_geotiff_tiles.py s2_10TET_20200828.tif -o tiles/s2_2020/ --png
python make_geotiff_tiles.py s2_10TET_20250720.tif -o tiles/s2_2025/ --png

# 5. Launch the viewer
python -m http.server 8000
# Open in browser:
#   http://localhost:8000/visualizer/interactive.html?tiles=../tiles/{z}/{x}/{y}.tif&left=../tiles/s2_2020/{z}/{x}/{y}.png&right=../tiles/s2_2025/{z}/{x}/{y}.png
```

### Using a different tile

Every script that previously hard-coded Seattle now accepts a `--tile` flag with
any Sentinel-2 MGRS tile ID (e.g. `33UUP`, `T32TQM`, `10SGD`):

```bash
python download_s2_scene.py    --tile 33UUP --year 2019 2024
python download_aef_geotiff.py --tile 33UUP --year 2019 2024
python pca_diff.py aef_33UUP_2019.tif aef_33UUP_2024.tif -n 3
python make_geotiff_tiles.py aef_diff_pca3_33UUP_2024_minus_33UUP_2019.tif -o tiles/
```

## How It Works

The pipeline downloads 64-band int8 AEF (AI Earth Foundation) embeddings from a
public Zarr v3 mosaic on S3, computes a per-pixel temporal diff, reduces it via
PCA, and tiles the result for browser consumption.

```
download_s2_scene.py        Fetch low-cloud S2 true-color scene per year
         │
download_aef_geotiff.py     Fetch 64-band embeddings for any year (2017–2025)
         │
         ▼
    pca_diff.py              Diff two years → PCA reduction
         │
         ▼
 make_geotiff_tiles.py       Slice GeoTIFF → {z}/{x}/{y}.tif or .png web-mercator pyramid
         │
         ▼
  visualizer/*.html          Leaflet side-by-side viewer with labeling + in-browser ML
```

`generate_aef_map.py` wraps the download + diff + PCA steps into a single script.

## Scripts

### `download_s2_scene.py`

Downloads the least-cloudy Sentinel-2 true-color (TCI) scene for a given MGRS
tile and year from the [Element 84 Earth Search](https://www.element84.com/earth-search/)
STAC catalog. Outputs a GeoTIFF that can be tiled with `make_geotiff_tiles.py --png`.

```bash
python download_s2_scene.py --tile T10TET --year 2020 2025
python download_s2_scene.py --tile 33UUP --year 2022 --max-cloud 5
python download_s2_scene.py --tile T10TET --year 2020 -o scenes/
```

### `download_aef_geotiff.py`

Downloads 64-band AEF embedding GeoTIFFs for any Sentinel-2 MGRS tile. Multiple
years download in parallel.

```bash
python download_aef_geotiff.py --tile T10TET                       # defaults: 2020 + 2025
python download_aef_geotiff.py --tile T10TET --year 2018 2022      # specific years
python download_aef_geotiff.py --tile 33UUP -y 2020 -o my_2020.tif # custom output
```

### `pca_diff.py`

Pixel-wise diff of two AEF GeoTIFFs → PCA reduction. Stores explained variance
in GeoTIFF metadata tags.

```bash
python pca_diff.py aef_10TET_2020.tif aef_10TET_2025.tif           # 8 components
python pca_diff.py aef_10TET_2020.tif aef_10TET_2025.tif -n 3      # 3 components
python pca_diff.py aef_10TET_2020.tif aef_10TET_2025.tif --subsample 500
```

### `make_geotiff_tiles.py`

Slices any GeoTIFF into a web-mercator `{z}/{x}/{y}.tif` (or `.png`) tile pyramid
using multiprocessing + shared memory.

```bash
python make_geotiff_tiles.py input.tif                               # zoom 8–14 GeoTIFF
python make_geotiff_tiles.py input.tif -o out/ --zoom-min 10 --zoom-max 16
python make_geotiff_tiles.py input.tif --png                         # RGB PNG tiles
python make_geotiff_tiles.py input.tif --workers 8                   # fewer CPUs
```

### `generate_aef_map.py`

All-in-one: downloads two years of AEF data for any tile, diffs, and outputs a
PCA-3 RGB GeoTIFF.

```bash
python generate_aef_map.py --tile T10TET
python generate_aef_map.py --tile 33UUP --year-a 2019 --year-b 2024 -o diff_33UUP.tif
```

### `evaluate_cv.py`

Evaluates change-detection performance using labeled points and AEF embedding
diffs. Fits PCA on the full raster diff, saves explained variance ratios, and
runs nested 10-fold cross-validation with logistic regression (inner 5-fold for
C hyperparameter selection).

```bash
python evaluate_cv.py aef_2020.tif aef_2024.tif data/labels.geojson
python evaluate_cv.py aef_2020.tif aef_2024.tif data/labels.geojson --variance-out ev.json
python evaluate_cv.py aef_2020.tif aef_2024.tif data/labels.geojson --folds 5
```

Tests three feature sets (3-band PCA, 8-band PCA, full 64-band diff) and reports
accuracy, F1, precision, recall, and best C per fold. See
[EVALUATION.md](EVALUATION.md) for detailed results.

## Data

The `data/` directory contains labeled point datasets (GeoJSON) exported from
the browser visualizer. Each file has Point features with a `label` property
(`"change"` or `"nochange"`), used as ground truth for `evaluate_cv.py`.

## Visualizer

The HTML viewers (`visualizer/`) are self-contained — no build step, no
bundler. They use Leaflet + GeoTIFF.js and load embedding tiles as
`{z}/{x}/{y}.tif` pyramids.

### Query Parameters

Both `index.html` and `interactive.html` accept these URL parameters:

| Parameter    | Description                                 | Default                            |
|-------------|---------------------------------------------|------------------------------------|
| `tiles`     | Embedding tile URL template                  | CDN tiles (Seattle)                |
| `left`      | Left (earlier) S2 imagery tile URL           | CDN 2020 Seattle                   |
| `right`     | Right (later) S2 imagery tile URL            | CDN 2024 Seattle                   |
| `leftLabel` | Label for the left imagery layer             | `2020-08-28`                       |
| `rightLabel`| Label for the right imagery layer            | `2024-07-20`                       |
| `lat`       | Initial map center latitude                  | `47.355`                           |
| `lng`       | Initial map center longitude                 | `-122.267`                         |
| `minZoom`   | Min zoom for embedding tiles                 | `8`                                |
| `maxZoom`   | Max native zoom for embedding tiles          | `14`                               |

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

Sentinel-2 true-color imagery is downloaded from the
[Element 84 Earth Search](https://www.element84.com/earth-search/) STAC catalog
(Sentinel-2 L2A collection) — no credentials needed.