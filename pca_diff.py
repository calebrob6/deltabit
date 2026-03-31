#!/usr/bin/env python3
"""
Compute the embedding diff between two AEF GeoTIFFs and reduce via PCA.

Outputs a multi-band float32 GeoTIFF of PCA components with explained
variance metadata.

Requirements:
    pip install numpy rasterio scikit-learn

Usage:
    python pca_diff.py aef_stlouis_2020.tif aef_stlouis_2025.tif
    python pca_diff.py aef_2020.tif aef_2025.tif -n 3 -o my_diff.tif
"""

import argparse
import json
import numpy as np
import rasterio
from sklearn.decomposition import PCA


def main():
    parser = argparse.ArgumentParser(
        description="Compute PCA of the embedding diff between two AEF GeoTIFFs",
    )
    parser.add_argument("file_a", help="Earlier-year AEF GeoTIFF (e.g. aef_stlouis_2020.tif)")
    parser.add_argument("file_b", help="Later-year AEF GeoTIFF (e.g. aef_stlouis_2025.tif)")
    parser.add_argument(
        "-n", "--n-components", type=int, default=8,
        help="Number of PCA components (default: 8)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output GeoTIFF path (default: auto-generated from inputs)",
    )
    parser.add_argument(
        "--subsample", type=int, default=1000,
        help="Subsample factor for PCA fitting (default: every 1000th pixel)",
    )
    args = parser.parse_args()

    if args.output is None:
        a_stem = args.file_a.replace(".tif", "").replace("aef_", "")
        b_stem = args.file_b.replace(".tif", "").replace("aef_", "")
        args.output = f"aef_diff_pca{args.n_components}_{b_stem}_minus_{a_stem}.tif"

    print(f"Reading {args.file_a}…")
    with rasterio.open(args.file_a) as src:
        d_a = src.read()
        profile = src.profile.copy()

    print(f"Reading {args.file_b}…")
    with rasterio.open(args.file_b) as src:
        d_b = src.read()

    print(f"Computing diff ({args.file_b} - {args.file_a})…")
    diff = d_b.astype(np.float32) - d_a.astype(np.float32)
    del d_a, d_b

    nbands, H, W = diff.shape
    print(f"Diff shape: {nbands} bands, {H}x{W}")

    pixels = diff.reshape(nbands, -1).T
    del diff

    valid_mask = np.all(np.isfinite(pixels), axis=1)
    n_valid = valid_mask.sum()
    print(f"Valid pixels: {n_valid:,} / {len(valid_mask):,}")

    valid_pixels = pixels[valid_mask]

    subsample = valid_pixels[:: args.subsample]
    print(f"Fitting PCA on {len(subsample):,} subsampled pixels (every {args.subsample}th)…")
    pca = PCA(n_components=args.n_components)
    pca.fit(subsample)
    del subsample

    evr = pca.explained_variance_ratio_
    total = float(evr.sum())
    print("\nExplained variance ratio per component:")
    for i, v in enumerate(evr):
        print(f"  PC{i+1}: {v:.6f}  ({v*100:.2f}%)")
    print(f"  Total (sum of {args.n_components}): {total:.6f}  ({total*100:.2f}%)")

    print("\nTransforming all pixels…")
    result = np.full((len(valid_mask), args.n_components), np.nan, dtype=np.float32)
    result[valid_mask] = pca.transform(valid_pixels).astype(np.float32)
    del valid_pixels

    result = result.T.reshape(args.n_components, H, W)

    profile.update(
        count=args.n_components, dtype="float32",
        compress="deflate", predictor=2,
    )
    print(f"Writing {args.output}…")
    with rasterio.open(args.output, "w", **profile) as dst:
        dst.write(result)
        dst.update_tags(
            explained_variance_ratio=json.dumps(evr.tolist()),
            explained_variance_total=f"{total:.8f}",
        )

    print("Done!")


if __name__ == "__main__":
    main()
