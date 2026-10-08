"""Flood post-processing of the hydrology GeoTIFFs.

From hydro/hand.tif and hydro/sink_depth.tif this step derives:
  * HAND inundation scenarios: for each water level h (m above the channel),
    cells with HAND <= h are flooded and the water depth there is h - HAND.
    This is the static "HAND flood mapping" approach (Nobre et al. 2016). Use it
    for screening; it is not a hydrodynamic simulation.
  * flood susceptibility classes from HAND breaks (1 = very low ... 5 = very high)
  * pluvial ponding zones: closed depressions deeper than a minimum depth
  * area statistics within the AOI (CSV + JSON), optional polygons (GeoPackage)
    and a quick-look PNG map.
"""
import csv
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask, shapes
from shapely.geometry import shape

from rio_utils import NODATA, write_raster

log = logging.getLogger(__name__)

CLASS_NAMES = {1: "very low", 2: "low", 3: "moderate", 4: "high", 5: "very high"}


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1, masked=True).astype("float32"), src.transform, src.crs


def _polygons(mask, transform, crs, attrs, min_area_m2=0.0):
    geoms = [shape(g) for g, v in shapes(mask.astype("uint8"), mask=mask, transform=transform) if v == 1]
    gdf = gpd.GeoDataFrame({**{k: [v] * len(geoms) for k, v in attrs.items()}}, geometry=geoms, crs=crs)
    gdf["area_m2"] = gdf.area
    return gdf[gdf["area_m2"] >= min_area_m2]


def susceptibility_classes(hand, breaks):
    """breaks [b1<b2<b3<b4]: HAND<=b1 -> 5 (very high) ... HAND>b4 -> 1 (very low); 0 = no data."""
    cls = np.zeros(hand.shape, dtype="uint8")
    valid = ~np.ma.getmaskarray(hand)
    h = hand.filled(np.inf)
    edges = [-np.inf] + list(breaks) + [np.inf]
    for i in range(len(edges) - 1):
        cls[valid & (h > edges[i]) & (h <= edges[i + 1])] = 5 - i
    return cls


def run_flood(hydro_dir, flood_dir, cfg, aoi_utm):
    hydro_dir, flood_dir = Path(hydro_dir), Path(flood_dir)
    flood_dir.mkdir(parents=True, exist_ok=True)
    hand, tr, crs = _read(hydro_dir / "hand.tif")
    sink, _, _ = _read(hydro_dir / "sink_depth.tif")
    cell_km2 = abs(tr.a * tr.e) / 1e6
    in_aoi = ~geometry_mask([aoi_utm], out_shape=hand.shape, transform=tr)
    valid = ~np.ma.getmaskarray(hand)
    aoi_km2 = float((in_aoi & valid).sum() * cell_km2)
    rows, vectors = [], []
    out = {}

    # 1. HAND inundation scenarios
    for h in cfg["water_levels_m"]:
        flooded = valid & (hand.filled(np.inf) <= h)
        depth = np.where(flooded, h - hand.filled(0), NODATA).astype("float32")
        p = write_raster(flood_dir / f"inundation_depth_{h:g}m.tif", depth, tr, crs)
        out[f"inundation_{h:g}m"] = str(p)
        area = float((flooded & in_aoi).sum() * cell_km2)
        mean_d = float(depth[flooded & in_aoi].mean()) if area else 0.0
        rows.append({"layer": f"inundation_{h:g}m", "class": f"HAND <= {h:g} m",
                     "area_km2": round(area, 4), "pct_of_aoi": round(100 * area / aoi_km2, 2),
                     "mean_depth_m": round(mean_d, 2)})
        if cfg.get("vectorize"):
            vectors.append(_polygons(flooded & in_aoi, tr, crs, {"layer": f"inundation_{h:g}m", "level_m": h}))

    # 2. Susceptibility classes
    cls = susceptibility_classes(hand, cfg["hand_class_breaks_m"])
    p = write_raster(flood_dir / "flood_susceptibility.tif", cls, tr, crs, nodata=0, dtype="uint8")
    out["susceptibility"] = str(p)
    b = cfg["hand_class_breaks_m"]
    ranges = {5: f"HAND <= {b[0]} m", 4: f"{b[0]}-{b[1]} m", 3: f"{b[1]}-{b[2]} m",
              2: f"{b[2]}-{b[3]} m", 1: f"> {b[3]} m"}
    for c in (5, 4, 3, 2, 1):
        area = float(((cls == c) & in_aoi).sum() * cell_km2)
        rows.append({"layer": "susceptibility", "class": f"{c} {CLASS_NAMES[c]} ({ranges[c]})",
                     "area_km2": round(area, 4), "pct_of_aoi": round(100 * area / aoi_km2, 2),
                     "mean_depth_m": ""})

    # 3. Pluvial ponding zones (closed depressions)
    pond = ~np.ma.getmaskarray(sink) & (sink.filled(0) >= cfg["ponding_min_depth_m"])
    pond_depth = np.where(pond, sink.filled(0), NODATA).astype("float32")
    p = write_raster(flood_dir / "ponding_depth.tif", pond_depth, tr, crs)
    out["ponding"] = str(p)
    area = float((pond & in_aoi).sum() * cell_km2)
    vol_m3 = float(sink.filled(0)[pond & in_aoi].sum() * abs(tr.a * tr.e))
    rows.append({"layer": "ponding", "class": f"sink depth >= {cfg['ponding_min_depth_m']} m",
                 "area_km2": round(area, 4), "pct_of_aoi": round(100 * area / aoi_km2, 2),
                 "mean_depth_m": round(float(sink.filled(0)[pond & in_aoi].mean()), 2) if area else 0.0})
    if cfg.get("vectorize"):
        vectors.append(_polygons(pond & in_aoi, tr, crs, {"layer": "ponding", "level_m": None},
                                 min_area_m2=cfg.get("ponding_min_area_m2", 0)))

    # 4. Statistics, polygons, quick-look
    with open(flood_dir / "flood_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"aoi_area_km2": round(aoi_km2, 3), "ponding_volume_m3": round(vol_m3), "rows": rows}
    (flood_dir / "flood_summary.json").write_text(json.dumps(summary, indent=2))
    out["summary_csv"] = str(flood_dir / "flood_summary.csv")

    if vectors:
        gpkg = flood_dir / "flood_polygons.gpkg"
        gpkg.unlink(missing_ok=True)
        for gdf in vectors:
            if len(gdf):
                gdf.to_file(gpkg, layer=gdf["layer"].iloc[0], driver="GPKG")
        out["polygons"] = str(gpkg)

    if cfg.get("quicklook", True):
        out["quicklook"] = str(quicklook(hydro_dir, flood_dir, cls, tr, aoi_utm,
                                         max(cfg["water_levels_m"])))

    for r in rows:
        log.info("Flood %-16s %-32s %9.3f km2 (%5.1f%%)", r["layer"], r["class"], r["area_km2"], r["pct_of_aoi"])
    return out, summary


def quicklook(hydro_dir, flood_dir, cls, tr, aoi_utm, level):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    hs, _, _ = _read(hydro_dir / "hillshade.tif")
    depth, _, _ = _read(flood_dir / f"inundation_depth_{level:g}m.tif")
    h, w = cls.shape
    ext = [tr.c, tr.c + w * tr.a, tr.f + h * tr.e, tr.f]
    cmap = ListedColormap(["#1a9850", "#a6d96a", "#fee08b", "#f46d43", "#a50026"])
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150, constrained_layout=True)
    polys = getattr(aoi_utm, "geoms", [aoi_utm])
    for ax in axs:
        ax.imshow(hs, cmap="gray", extent=ext)
        for poly in polys:
            ax.plot(*poly.exterior.xy, color="black", lw=1)
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].imshow(np.ma.masked_equal(cls, 0), cmap=cmap, vmin=0.5, vmax=5.5, alpha=0.7, extent=ext)
    axs[0].legend(handles=[Patch(color=cmap(i), label=CLASS_NAMES[i + 1].capitalize()) for i in range(4, -1, -1)],
                  loc="lower left", fontsize=8, title="Flood susceptibility", title_fontsize=8)
    axs[0].set_title("Flood susceptibility (HAND classes)")
    im = axs[1].imshow(depth, cmap="Blues", vmin=0, vmax=level, alpha=0.85, extent=ext)
    fig.colorbar(im, ax=axs[1], shrink=0.8, label="Water depth (m)")
    axs[1].set_title(f"HAND inundation, water level {level:g} m above channel")
    png = flood_dir / "flood_quicklook.png"
    fig.savefig(png, facecolor="white")
    plt.close(fig)
    return png
