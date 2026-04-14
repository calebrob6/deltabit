#!/usr/bin/env python3
"""
Evaluate change-detection performance using AEF embedding diffs at labeled points.

Fits PCA on the full raster diff (subsampled), saves explained variance ratios,
then runs 10-fold cross-validation with logistic regression for 3-band, 8-band,
and full 64-band diff features.

Requirements:
    pip install numpy rasterio scikit-learn

Usage:
    python evaluate_cv.py aef_2020.tif aef_2024.tif labels.geojson
    python evaluate_cv.py aef_2020.tif aef_2024.tif labels.geojson --variance-out ev.json
"""

import argparse
import json
import numpy as np
import rasterio
from rasterio.transform import rowcol
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_labels(geojson_path):
    """Load labeled points from GeoJSON. Returns (coords, labels)."""
    with open(geojson_path) as f:
        data = json.load(f)

    coords = []
    labels = []
    for feat in data["features"]:
        lon, lat = feat["geometry"]["coordinates"][:2]
        label = feat["properties"]["label"]
        coords.append((lon, lat))
        labels.append(1 if label == "change" else 0)

    return np.array(coords), np.array(labels)


def sample_pixels(raster_path, coords):
    """Sample pixel values at lon/lat coordinates. Returns (n_points, n_bands)."""
    with rasterio.open(raster_path) as src:
        transform = src.transform
        data = src.read()

    n_bands, height, width = data.shape
    values = []
    valid = []

    for i, (lon, lat) in enumerate(coords):
        row, col = rowcol(transform, lon, lat)
        if 0 <= row < height and 0 <= col < width:
            values.append(data[:, row, col])
            valid.append(i)
        else:
            values.append(np.full(n_bands, np.nan))
            valid.append(i)

    return np.array(values, dtype=np.float32), valid


def fit_full_pca(file_a, file_b, subsample=1000):
    """Fit PCA with all 64 components on the raster diff (subsampled)."""
    print(f"Reading {file_a}...")
    with rasterio.open(file_a) as src:
        d_a = src.read()

    print(f"Reading {file_b}...")
    with rasterio.open(file_b) as src:
        d_b = src.read()

    n_bands = d_a.shape[0]
    print(f"Computing diff ({n_bands} bands)...")
    diff = d_b.astype(np.float32) - d_a.astype(np.float32)
    del d_a, d_b

    pixels = diff.reshape(n_bands, -1).T
    del diff

    valid_mask = np.all(np.isfinite(pixels), axis=1)
    valid_pixels = pixels[valid_mask]
    del pixels

    subsample_pixels = valid_pixels[::subsample]
    print(f"Fitting PCA ({n_bands} components) on {len(subsample_pixels):,} subsampled pixels...")
    pca = PCA(n_components=n_bands)
    pca.fit(subsample_pixels)
    del subsample_pixels, valid_pixels

    return pca


def run_cv(X, y, n_components, pca, n_folds=10):
    """Run nested CV with C sweep. Outer folds evaluate, inner folds tune C."""
    if n_components < X.shape[1]:
        X_proj = X @ pca.components_[:n_components].T
    else:
        X_proj = X

    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]),
        param_grid={"clf__C": C_values},
        cv=inner_cv,
        scoring="f1",
        refit=True,
    )

    outer_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1", "precision", "recall"]

    # Manual outer loop to capture best C per fold
    fold_results = {m: [] for m in scoring}
    best_Cs = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_proj, y)):
        X_train, X_test = X_proj[train_idx], X_proj[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        grid.fit(X_train, y_train)
        best_C = grid.best_params_["clf__C"]
        best_Cs.append(best_C)

        y_pred = grid.predict(X_test)
        fold_results["accuracy"].append(float((y_pred == y_test).mean()))
        fold_results["f1"].append(float(f1_score(y_test, y_pred)))
        fold_results["precision"].append(float(precision_score(y_test, y_pred)))
        fold_results["recall"].append(float(recall_score(y_test, y_pred)))

    metrics = {
        metric: {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
        for metric, vals in fold_results.items()
    }

    return metrics, best_Cs


def main():
    parser = argparse.ArgumentParser(
        description="10-fold CV evaluation of change detection with AEF embeddings",
    )
    parser.add_argument("file_a", help="Earlier-year 64-band int8 AEF GeoTIFF")
    parser.add_argument("file_b", help="Later-year 64-band int8 AEF GeoTIFF")
    parser.add_argument("labels", help="GeoJSON file with labeled points")
    parser.add_argument(
        "--variance-out",
        default="explained_variance.json",
        help="Output path for explained variance ratios (default: explained_variance.json)",
    )
    parser.add_argument(
        "--subsample", type=int, default=1000,
        help="Subsample factor for PCA fitting (default: every 1000th pixel)",
    )
    parser.add_argument(
        "--folds", type=int, default=10,
        help="Number of CV folds (default: 10)",
    )
    args = parser.parse_args()

    # Load labeled points
    print("Loading labels...")
    coords, labels = load_labels(args.labels)
    print(f"  {len(labels)} points: {(labels == 1).sum()} change, {(labels == 0).sum()} nochange")

    # Sample pixels from both rasters
    print("Sampling pixel values...")
    vals_a, _ = sample_pixels(args.file_a, coords)
    vals_b, _ = sample_pixels(args.file_b, coords)

    # Compute diff at labeled points
    diff_points = vals_b.astype(np.float32) - vals_a.astype(np.float32)
    del vals_a, vals_b

    # Drop any points with NaN
    valid = np.all(np.isfinite(diff_points), axis=1)
    if valid.sum() < len(valid):
        print(f"  Dropping {(~valid).sum()} points with NaN values")
    diff_points = diff_points[valid]
    labels = labels[valid]
    print(f"  Using {len(labels)} valid points")

    # Fit PCA on full raster diff
    pca = fit_full_pca(args.file_a, args.file_b, subsample=args.subsample)

    # Save explained variance
    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)
    variance_data = {
        "n_components": len(evr),
        "explained_variance_ratio": evr.tolist(),
        "cumulative_variance_ratio": cumulative.tolist(),
        "summary": {
            "3_components": float(cumulative[2]),
            "8_components": float(cumulative[7]),
            "64_components": float(cumulative[-1]),
        },
    }

    print(f"\nSaving explained variance to {args.variance_out}...")
    with open(args.variance_out, "w") as f:
        json.dump(variance_data, f, indent=2)

    print("\nExplained variance summary:")
    print(f"  3 components: {cumulative[2]*100:.2f}%")
    print(f"  8 components: {cumulative[7]*100:.2f}%")
    print(f"  64 components: {cumulative[-1]*100:.2f}%")

    # Run CV experiments
    band_configs = [
        ("3-band (PCA-3)", 3),
        ("8-band (PCA-8)", 8),
        ("64-band (full diff)", 64),
    ]

    all_results = {}
    for name, n in band_configs:
        print(f"\n{'='*60}")
        print(f"  {args.folds}-fold nested CV: {name}")
        print(f"  (inner 5-fold for C selection, C ∈ [0.001..1000])")
        print(f"{'='*60}")

        results, best_Cs = run_cv(diff_points, labels, n, pca, n_folds=args.folds)
        all_results[name] = {"metrics": results, "best_Cs": best_Cs}

        for metric, vals in results.items():
            print(f"  {metric:>12s}: {vals['mean']:.4f} ± {vals['std']:.4f}")
        print(f"  {'best C/fold':>12s}: {best_Cs}")

    # Summary table
    print(f"\n{'='*60}")
    print("  Summary (nested CV with C tuning)")
    print(f"{'='*60}")
    print(f"  {'Features':<22s} {'Accuracy':>12s} {'F1':>12s} {'Median C':>10s}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*10}")
    for name, data in all_results.items():
        r = data["metrics"]
        acc, f1 = r["accuracy"], r["f1"]
        median_c = float(np.median(data["best_Cs"]))
        print(f"  {name:<22s} {acc['mean']:.4f}±{acc['std']:.4f} {f1['mean']:.4f}±{f1['std']:.4f} {median_c:>10g}")

    print("\nDone!")


if __name__ == "__main__":
    main()
