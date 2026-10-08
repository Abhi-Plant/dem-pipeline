"""QA: per-DEM summaries plus pairwise differences on the shared grid (e.g. AWS vs GEE)."""
from itertools import combinations
from pathlib import Path

import numpy as np
import rasterio

from rio_utils import raster_summary


def compare(a_path, b_path):
    with rasterio.open(a_path) as a, rasterio.open(b_path) as b:
        if (a.shape, a.transform, a.crs) != (b.shape, b.transform, b.crs):
            return {"error": "grids differ; cannot compare pixel-wise"}
        A = a.read(1, masked=True).astype("float64")
        B = b.read(1, masked=True).astype("float64")
    d = (A - B).compressed()
    if d.size == 0:
        return {"error": "no overlapping valid pixels"}
    ad = np.abs(d)
    return {
        "a": Path(a_path).name, "b": Path(b_path).name, "n_pixels": int(d.size),
        "mean_diff": float(d.mean()), "mae": float(ad.mean()),
        "rmse": float(np.sqrt((d ** 2).mean())), "p95_abs": float(np.percentile(ad, 95)),
        "max_abs": float(ad.max()),
    }


def run_qa(dem_paths):
    """dem_paths: {label: path}. Returns summaries and all pairwise comparisons."""
    report = {"summaries": {k: raster_summary(v) for k, v in dem_paths.items()},
              "comparisons": []}
    for (ka, pa), (kb, pb) in combinations(dem_paths.items(), 2):
        report["comparisons"].append({"pair": f"{ka} - {kb}", **compare(pa, pb)})
    return report
