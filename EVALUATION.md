# Deltabit: AEF Change Detection Evaluation

## Overview

This report evaluates how well [AEF (AI Earth Foundation)](https://source.coop/tge-labs/aef-mosaic)
embedding differences can distinguish "change" from "no-change" pixels in
satellite imagery, using logistic regression with 10-fold nested
cross-validation across three feature dimensionalities.

## Study Area

| Property | Value |
|---|---|
| Sentinel-2 MGRS tile | **T10TET** |
| Region | Seattle metro area, Washington, USA |
| Bounding box (EPSG:4326) | 46.857°N – 47.854°N, 123.000°W – 121.533°W |
| Comparison years | **2020** vs **2024** |
| Spatial resolution | 10 m (Sentinel-2 native grid) |
| Raster dimensions | 11,099 × 16,337 pixels (181.3 M pixels) |

## Data Source

Embeddings are drawn from the **AEF Mosaic**, a global Zarr v3 archive on S3
(`s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic`). Each pixel is a
64-dimensional signed int8 vector encoding land-surface characteristics derived
from Sentinel-2 imagery. The mosaic covers 2017–2025 with annual composites.

## Labels

| Property | Value |
|---|---|
| File | `deltabit-labels-2026-04-14T03-56-20.geojson` |
| Total points | 336 |
| Change | 190 (56.5%) |
| No change | 146 (43.5%) |
| Spatial extent | 47.35°N – 47.78°N, 122.35°W – 121.66°W |
| Labeling method | Manual pixel labeling in the Deltabit browser viewer |

Labels were created by visually comparing 2020 and 2024 Sentinel-2 true-color
imagery using the Leaflet side-by-side swipe interface. Each labeled point is a
single 10 m pixel classified as either "change" or "nochange".

## Feature Construction

For each labeled point, the 64-band int8 embedding vectors for 2020 and 2024 are
sampled from the AEF GeoTIFFs. The **per-pixel temporal difference** (2024 − 2020)
produces a 64-dimensional float32 feature vector capturing what changed at that
location.

Three feature sets are evaluated:

| Feature set | Dimensions | Construction |
|---|---|---|
| **PCA-3** | 3 | Project 64-d diff onto first 3 principal components |
| **PCA-8** | 8 | Project 64-d diff onto first 8 principal components |
| **Full diff** | 64 | Raw 64-d embedding difference (no PCA) |

PCA is fit on a subsample of the full raster diff (every 1,000th pixel out of
181.3 M, yielding ~181 K training pixels for the PCA fit). The same PCA
projection is then applied to the labeled points.

## Explained Variance

The 64 PCA components account for the following cumulative variance in the
raster-wide embedding diff:

| Components | Cumulative variance |
|---|---|
| 1 | 18.25% |
| 2 | 25.10% |
| 3 | **30.89%** |
| 4 | 34.80% |
| 5 | 38.60% |
| 8 | **47.75%** |
| 9 | 50.37% |
| 23 | 75.16% |
| 40 | 90.12% |
| 49 | 95.27% |
| 64 | 100.00% |

The variance is distributed broadly — no single component dominates. PC1
captures 18.3%, but 9 components are needed to reach 50%, and 40 for 90%.

### Per-component explained variance ratio

| PC | Var (%) | Cum (%) | | PC | Var (%) | Cum (%) |
|----|---------|---------|---|----|---------|---------|
| 1 | 18.25 | 18.25 | | 9 | 2.62 | 50.37 |
| 2 | 6.85 | 25.10 | | 10 | 2.38 | 52.75 |
| 3 | 5.79 | 30.89 | | 16 | 1.77 | 65.16 |
| 4 | 3.90 | 34.80 | | 32 | 0.84 | 84.14 |
| 5 | 3.80 | 38.60 | | 48 | 0.52 | 94.23 |
| 6 | 3.38 | 41.98 | | 56 | 0.28 | 98.14 |
| 7 | 2.90 | 44.88 | | 64 | 0.12 | 100.00 |
| 8 | 2.87 | 47.75 | | | | |

## Cross-Validation Experiment

### Method

- **Model**: Logistic regression (`sklearn.linear_model.LogisticRegression`,
  solver=`lbfgs`, max_iter=1000)
- **Preprocessing**: Per-feature z-score standardization (`StandardScaler`)
- **Evaluation**: 10-fold stratified nested cross-validation
  - **Outer loop**: 10-fold stratified CV for unbiased performance estimation
  - **Inner loop**: 5-fold stratified CV within each training fold for
    regularization parameter selection
- **Hyperparameter sweep**: C ∈ {0.001, 0.01, 0.1, 1, 10, 100, 1000}
  (inverse regularization strength)
- **Inner scoring**: F1 score for C selection
- **Random state**: 42 (deterministic splits)

### Results

| Feature set | Accuracy | F1 | Precision | Recall | Median C |
|---|---|---|---|---|---|
| **PCA-3** (3-d) | 0.899 ± 0.052 | 0.902 ± 0.054 | 0.973 ± 0.037 | 0.847 ± 0.098 | 55 |
| **PCA-8** (8-d) | 0.959 ± 0.042 | 0.962 ± 0.039 | 0.984 ± 0.034 | 0.942 ± 0.055 | 10 |
| **Full diff** (64-d) | 0.988 ± 0.015 | 0.989 ± 0.013 | 1.000 ± 0.000 | 0.979 ± 0.026 | 5.5 |

### Best C per fold

| Fold | PCA-3 | PCA-8 | Full 64-d |
|------|-------|-------|-----------|
| 1 | 100 | 10 | 100 |
| 2 | 100 | 10 | 0.1 |
| 3 | 1 | 100 | 10 |
| 4 | 10 | 1 | 1 |
| 5 | 10 | 100 | 1 |
| 6 | 1 | 10 | 10 |
| 7 | 100 | 10 | 1 |
| 8 | 1000 | 1 | 10 |
| 9 | 10 | 10 | 10 |
| 10 | 100 | 100 | 0.1 |

### Key Findings

1. **More dimensions help significantly.** Going from 3 → 8 PCA components
   improves accuracy by 6 points (0.899 → 0.959) and recall by nearly 10 points
   (0.847 → 0.942). The full 64-d diff pushes accuracy to 0.988.

2. **The initial run with default C=1 underestimated the gap.** The previous
   experiment (without C tuning) showed 8-band and 64-band performing
   identically at ~0.934 accuracy. With proper C tuning, the full 64-d features
   clearly outperform the 8-band PCA projection (0.988 vs 0.959).

3. **Lower-dimensional features need stronger C (weaker regularization).**
   PCA-3 selects high C values (median 55), while the full 64-d diff prefers
   moderate regularization (median 5.5). This makes sense: with fewer features,
   the model needs more freedom to fit the limited signal; with 64 features,
   regularization prevents overfitting.

4. **Precision is consistently very high (≥0.97).** The model rarely
   misclassifies no-change pixels as change. The main error mode is missed
   changes (lower recall), which improves from 0.847 to 0.979 as feature
   dimensionality increases.

5. **Variance in the embedding diff is broadly distributed, but change
   detection signal is concentrated.** Although 8 PCA components capture only
   48% of the total variance, they achieve 96% accuracy. The remaining 52% of
   variance in components 9–64 contributes an additional 3% accuracy,
   suggesting that some change-relevant signal exists in the lower-variance
   components.

## Reproducibility

```bash
conda env create -f environment.yml
conda activate in-browser-embedding

python evaluate_cv.py \
    aef_diff_pca3_seattle_aef_2020.tif \
    aef_diff_pca3_seattle_aef_2024.tif \
    deltabit-labels-2026-04-14T03-56-20.geojson
```
