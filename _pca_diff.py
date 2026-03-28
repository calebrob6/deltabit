import numpy as np
import rasterio
from sklearn.decomposition import PCA
import json

f2020 = 'aef_diff_pca3_seattle_aef_2020.tif'
f2024 = 'aef_diff_pca3_seattle_aef_2024.tif'
out_tif = 'aef_diff_pca8_2024_minus_2020.tif'

print("Reading 2020…")
with rasterio.open(f2020) as s20:
    d20 = s20.read()
    profile = s20.profile.copy()

print("Reading 2024…")
with rasterio.open(f2024) as s24:
    d24 = s24.read()

print("Computing diff (2024 - 2020)…")
diff = (d24.astype(np.float32) - d20.astype(np.float32))
del d20, d24

nbands, H, W = diff.shape
print(f"Diff shape: {nbands} bands, {H}x{W}")

# Reshape to (n_pixels, 64)
pixels = diff.reshape(nbands, -1).T
del diff

# Valid pixel mask
valid_mask = np.all(np.isfinite(pixels), axis=1)
n_valid = valid_mask.sum()
print(f"Valid pixels: {n_valid:,} / {len(valid_mask):,}")

valid_pixels = pixels[valid_mask]

# Fit PCA on every 1000th valid pixel
subsample = valid_pixels[::1000]
print(f"Fitting PCA on {len(subsample):,} subsampled pixels (every 1000th)…")
pca = PCA(n_components=8)
pca.fit(subsample)
del subsample

evr = pca.explained_variance_ratio_
total = float(evr.sum())
print("\nExplained variance ratio per component:")
for i, v in enumerate(evr):
    print(f"  PC{i+1}: {v:.6f}  ({v*100:.2f}%)")
print(f"  Total (sum of 8): {total:.6f}  ({total*100:.2f}%)")

# Transform all valid pixels
print("\nTransforming all pixels…")
result = np.full((len(valid_mask), 8), np.nan, dtype=np.float32)
result[valid_mask] = pca.transform(valid_pixels).astype(np.float32)
del valid_pixels

result = result.T.reshape(8, H, W)

# Save GeoTIFF
profile.update(count=8, dtype='float32', compress='deflate', predictor=2)
print(f"Writing {out_tif}…")
with rasterio.open(out_tif, 'w', **profile) as dst:
    dst.write(result)
    dst.update_tags(
        explained_variance_ratio=json.dumps(evr.tolist()),
        explained_variance_total=f"{total:.8f}"
    )

print("Done!")
