"""Hydrological conditioning and flood/drainage derivatives with WhiteboxTools.

Workflow (D8, 30 m UTM grid):
  1. fill single-cell pits           -> removes speckle sinks
  2. least-cost depression breaching -> cuts through road/bridge "dams" in the
     DSM rather than filling them, which keeps the terrain realistic
  3. D8 pointer + flow accumulation  -> drainage network and contributing area
  4. streams (area threshold), Strahler order, stream vector
  5. HAND  (height above nearest drainage) -> main flood-susceptibility index
  6. TWI   (topographic wetness index)     -> where saturation and ponding occur
  7. sink depth on the raw DEM             -> closed depressions that store pluvial runoff
  8. slope, hillshade, optional watershed(s) from pour points
"""
import logging
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask

from rio_utils import write_raster

log = logging.getLogger(__name__)


def _wbt(work_dir):
    try:
        from whitebox import WhiteboxTools
    except ImportError as exc:
        raise ImportError("pip install whitebox  (WhiteboxTools Python frontend)") from exc
    wbt = WhiteboxTools()
    wbt.set_working_dir(str(work_dir))
    wbt.set_verbose_mode(False)
    return wbt


def _run(name, fn, *args, **kwargs):
    """Run a Whitebox tool. A Rust panic can still return 0, so also check that the output
    (by convention the last positional argument here) was written."""
    log.info("  whitebox: %s", name)
    out = Path(args[-1])
    out.unlink(missing_ok=True)
    rc = fn(*args, **kwargs)
    if rc != 0 or not out.exists():
        raise RuntimeError(f"WhiteboxTools '{name}' failed (rc={rc}); "
                           "set wbt.set_verbose_mode(True) in hydro._wbt to see the tool log")


def _set_vector_crs(path, crs):
    """WhiteboxTools may write shapefiles without a .prj file; add the CRS and save as GeoPackage."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    gpkg = Path(path).with_suffix(".gpkg")
    gdf.to_file(gpkg, driver="GPKG")
    return gpkg


def run_hydro(dem_path, hydro_dir, cfg, grid, aoi_utm=None):
    hydro_dir = Path(hydro_dir)
    hydro_dir.mkdir(parents=True, exist_ok=True)
    wbt = _wbt(hydro_dir)
    p = {k: str(hydro_dir / f"{k}.tif") for k in (
        "dem_pitfilled", "dem_cond", "d8_pointer", "flow_acc_cells", "sca",
        "streams", "stream_order", "slope_deg", "hand", "twi", "sink_depth", "hillshade")}
    dem = str(Path(dem_path).resolve())

    thr_cells = max(1, round(cfg["stream_threshold_km2"] * 1e6 / grid.res ** 2))
    log.info("Hydro: DEM=%s, stream threshold=%s km2 (%d cells)",
             dem, cfg["stream_threshold_km2"], thr_cells)

    _run("FillSingleCellPits", wbt.fill_single_cell_pits, dem, p["dem_pitfilled"])
    _run("BreachDepressionsLeastCost", wbt.breach_depressions_least_cost,
         p["dem_pitfilled"], p["dem_cond"], dist=int(cfg["breach_dist_cells"]), fill=True)
    _run("D8Pointer", wbt.d8_pointer, p["dem_cond"], p["d8_pointer"])
    _run("D8FlowAccumulation(cells)", wbt.d8_flow_accumulation,
         p["dem_cond"], p["flow_acc_cells"], out_type="cells")
    _run("D8FlowAccumulation(SCA)", wbt.d8_flow_accumulation,
         p["dem_cond"], p["sca"], out_type="specific contributing area")
    _run("ExtractStreams", wbt.extract_streams,
         p["flow_acc_cells"], p["streams"], threshold=thr_cells, zero_background=False)
    _run("StrahlerStreamOrder", wbt.strahler_stream_order,
         p["d8_pointer"], p["streams"], p["stream_order"])
    streams_shp = str(hydro_dir / "streams.shp")
    _run("RasterStreamsToVector", wbt.raster_streams_to_vector,
         p["streams"], p["d8_pointer"], streams_shp)
    _set_vector_crs(streams_shp, grid.crs)

    _run("Slope", wbt.slope, p["dem_cond"], p["slope_deg"], units="degrees")
    _run("ElevationAboveStream (HAND)", wbt.elevation_above_stream,
         p["dem_cond"], p["streams"], p["hand"])
    _run("WetnessIndex (TWI)", wbt.wetness_index, p["sca"], p["slope_deg"], p["twi"])
    _run("DepthInSink", wbt.depth_in_sink, dem, p["sink_depth"], zero_background=True)
    _run("Hillshade", wbt.hillshade, p["dem_cond"], p["hillshade"])

    if cfg.get("pour_points"):
        pts = gpd.read_file(cfg["pour_points"]).to_crs(grid.crs)
        pts_shp = hydro_dir / "pour_points_utm.shp"
        pts.to_file(pts_shp)
        snapped = str(hydro_dir / "pour_points_snapped.shp")
        _run("SnapPourPoints", wbt.snap_pour_points,
             str(pts_shp), p["flow_acc_cells"], snapped, snap_dist=cfg.get("snap_dist_m", 90))
        ws = str(hydro_dir / "watersheds.tif")
        _run("Watershed", wbt.watershed, p["d8_pointer"], snapped, ws)
        p["watersheds"] = ws
        ws_shp = str(hydro_dir / "watersheds.shp")
        _run("RasterToVectorPolygons", wbt.raster_to_vector_polygons, ws, ws_shp)
        _set_vector_crs(ws_shp, grid.crs)

    if cfg.get("clip_to_aoi") and aoi_utm is not None:
        clip_dir = hydro_dir / "aoi_clip"
        clip_dir.mkdir(exist_ok=True)
        for key in ("dem_cond", "hand", "twi", "slope_deg", "sink_depth", "flow_acc_cells"):
            clip_raster(p[key], aoi_utm, clip_dir / f"{key}_aoi.tif")
        log.info("Hydro: AOI-clipped rasters -> %s", clip_dir)

    return p


def clip_raster(src_path, geom, dst_path):
    with rasterio.open(src_path) as src:
        arr, tr = rio_mask(src, [geom], crop=True, nodata=src.nodata)
        write_raster(dst_path, arr[0], tr, src.crs, nodata=src.nodata)
