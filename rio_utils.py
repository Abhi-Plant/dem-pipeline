"""Small raster I/O helpers shared by every pipeline step."""
from pathlib import Path

import numpy as np
import rasterio

NODATA = -9999.0

GTIFF_PROFILE = dict(
    driver="GTiff",
    dtype="float32",
    count=1,
    nodata=NODATA,
    # No PREDICTOR=3: WhiteboxTools cannot read floating-point predictors.
    compress="deflate",
    tiled=True,
    blockxsize=256,
    blockysize=256,
    BIGTIFF="IF_SAFER",
)


def write_raster(path, arr, transform, crs, nodata=NODATA):
    """Write a single-band float32 GeoTIFF (deflate-compressed and tiled)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        **GTIFF_PROFILE,
        "height": arr.shape[0],
        "width": arr.shape[1],
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)
    return path


def raster_summary(path):
    """Basic descriptive stats for one raster, as a JSON-serialisable dict."""
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype("float64")
        valid = arr.compressed()
        return {
            "path": str(path),
            "crs": src.crs.to_string() if src.crs else None,
            "res": list(src.res),
            "shape": [src.height, src.width],
            "nodata_pct": round(100.0 * (1 - valid.size / arr.size), 3),
            "min": float(valid.min()) if valid.size else None,
            "max": float(valid.max()) if valid.size else None,
            "mean": float(valid.mean()) if valid.size else None,
            "std": float(valid.std()) if valid.size else None,
        }
