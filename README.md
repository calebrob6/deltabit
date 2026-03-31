# In-Browser Embedding Viewer

Interactive change-detection tool that runs entirely in the browser. Users swipe
between two dates of Sentinel-2 imagery, label pixels as "change" or "no change",
train a logistic regression model on AEF embeddings, and see predictions rendered
in real time via WebGPU (or CPU fallback).

## Repository Layout

```
├── download_aef_geotiff.py      # Download AEF embedding GeoTIFFs from the mosaic
├── pca_diff.py                  # PCA reduction of the embedding diff between two years
├── make_geotiff_tiles.py        # Slice any GeoTIFF into a {z}/{x}/{y} tile pyramid
├── generate_aef_map.py          # End-to-end: download + diff + PCA-3 RGB in one script
├── visualizer/
│   ├── interactive.html          # Seattle viewer (2020 vs 2024)
│   └── interactive_stlouis.html  # St. Louis viewer (2020 vs 2025)
├── aef_stlouis_2020.tif          # 64-band AEF embeddings, St. Louis, 2020
├── aef_stlouis_2025.tif          # 64-band AEF embeddings, St. Louis, 2025
├── T15SYC_20200712T164849_TCI_10m.tif  # Sentinel-2 true-color, 2020-07-12
├── T15SYC_20250701T164921_TCI_10m.tif  # Sentinel-2 true-color, 2025-07-01
├── s2_2020-07-12_stlouis_tiles/  # Pre-built Sentinel-2 PNG tiles (2020)
├── s2_2025-07-01_stlouis_tiles/  # Pre-built Sentinel-2 PNG tiles (2025)
└── aef_index.gpkg                # AEF tile index GeoPackage
```

## Setup

```bash
pip install numpy xarray zarr rasterio affine pyproj s3fs rasterix \
            scikit-learn mercantile
```

## Scripts

### `download_aef_geotiff.py` — Download AEF Embeddings

Downloads 64-band AEF embedding GeoTIFFs from the public S3-hosted mosaic for the
St. Louis area (Sentinel-2 tile T15SYC). Downloads multiple years in parallel.

```bash
# Download defaults (2020 and 2025)
python download_aef_geotiff.py

# Download specific years
python download_aef_geotiff.py --year 2018 2022

# Custom output path (single year only)
python download_aef_geotiff.py --year 2020 --output my_aef_2020.tif
```

### `pca_diff.py` — PCA of Embedding Diff

Computes the pixel-wise difference between two AEF GeoTIFFs and reduces the
64-band diff to a smaller number of PCA components. Stores explained variance
metadata in the output GeoTIFF tags.

```bash
# Default: 8 components, auto-named output
python pca_diff.py aef_stlouis_2020.tif aef_stlouis_2025.tif

# Custom component count and output
python pca_diff.py aef_stlouis_2020.tif aef_stlouis_2025.tif -n 3 -o diff_pca3.tif

# Adjust PCA subsample rate
python pca_diff.py aef_stlouis_2020.tif aef_stlouis_2025.tif --subsample 500
```

### `make_geotiff_tiles.py` — Tile Pyramid Generator

Slices any GeoTIFF into a web-mercator `{z}/{x}/{y}.tif` tile pyramid using
multiprocessing and shared memory for speed.

```bash
# Default: zoom 8–14, output to tiles/
python make_geotiff_tiles.py input.tif

# Custom zoom range and output directory
python make_geotiff_tiles.py input.tif -o my_tiles/ --zoom-min 10 --zoom-max 16

# Fewer workers for constrained environments
python make_geotiff_tiles.py input.tif --workers 8
```

### `generate_aef_map.py` — End-to-End Diff Map

All-in-one script that downloads two years of AEF data, computes the diff, and
produces a PCA-3 RGB GeoTIFF. Currently configured for the Seattle area.

```bash
python generate_aef_map.py --output aef_diff_pca3.tif
```

## Visualizer

The interactive HTML viewers use Leaflet with a side-by-side swipe control to
compare Sentinel-2 imagery from two dates. Users can:

1. **Swipe** between the two dates to visually inspect changes
2. **Label** pixels as "change" or "no change" by clicking on the map
3. **Train** a logistic regression model on the labelled AEF embeddings
4. **Predict** across all loaded tiles and visualize results as a heatmap or
   binary overlay (via WebGPU or CPU)

Open `visualizer/interactive_stlouis.html` in a browser (served via a web
server, or pass an embedding tile URL as a `?tiles=` query parameter).

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

## End-to-End Workflow

```bash
# 1. Download AEF embeddings for 2020 and 2025
python download_aef_geotiff.py

# 2. Compute PCA diff
python pca_diff.py aef_stlouis_2020.tif aef_stlouis_2025.tif

# 3. Generate tile pyramid for the embedding diff
python make_geotiff_tiles.py aef_diff_pca8_stlouis_2025_minus_stlouis_2020.tif -o diff_tiles/

# 4. Open the viewer (serve the repo directory with any HTTP server)
#    Then visit: http://localhost:8000/visualizer/interactive_stlouis.html?tiles=../diff_tiles/{z}/{x}/{y}.tif
python -m http.server 8000
```