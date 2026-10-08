"""Copernicus GLO-30 DEM pipeline for flood and drainage analysis.

Usage
-----
    python run_pipeline.py --config config.yaml
    python run_pipeline.py --config config.yaml --steps aws hydro
    python run_pipeline.py --config config.yaml --bbox 77.55 12.90 77.65 13.00 --name blr
    python run_pipeline.py --import-dem "C:/Downloads/dem_fabdem_gee_utm*.tif" --import-name fabdem_drive                            --hydro-input fabdem_drive --steps qa hydro flood

Output layout  (<output_dir>/<project_name>/)
    aoi.geojson, aoi_buffered.geojson, grid.json
    raw/   native 1-arcsec mosaic (EPSG:4326) and optional cached tiles
    dem/   DEMs on the common UTM analysis grid  (dem_<dataset>_<source>_utm.tif)
    hydro/ conditioned DEM, D8, flow accumulation, streams, HAND, TWI, sinks, ...
    flood/ HAND inundation scenarios, susceptibility classes, ponding, statistics
    qa/    qa_report.json
    run_report.json
"""
import argparse
import glob
import json
import logging
import sys
import time
from pathlib import Path

import yaml

from aoi import WGS84, build_grid, load_aoi, save_geojson
from rio_utils import raster_summary

ALL_STEPS = ["aws", "gee", "qa", "hydro", "flood"]
log = logging.getLogger("dem_pipeline")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--steps", nargs="+", choices=ALL_STEPS,
                    help="Steps to run (default: every step enabled in the config)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                    help="Override the AOI with a lon/lat bounding box")
    ap.add_argument("--name", help="Override project_name")
    ap.add_argument("--import-dem", nargs="+", metavar="TIF",
                    help="External DEM GeoTIFF(s) or glob patterns (e.g. GEE Drive export tiles) "
                         "to place on the analysis grid as dem/dem_<import-name>_utm.tif")
    ap.add_argument("--import-name", default="imported", help="Label for --import-dem (default: imported)")
    ap.add_argument("--hydro-input", help="Override hydro.input (e.g. fabdem_gee or an --import-name)")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    if args.bbox:
        cfg["aoi"].update(type="bbox", bbox=args.bbox)
    if args.name:
        cfg["project_name"] = args.name
    if args.hydro_input:
        cfg["hydro"]["input"] = args.hydro_input

    out_root = (cfg_path.parent / cfg["output_dir"] / cfg["project_name"]).resolve()
    dirs = {k: out_root / k for k in ("raw", "dem", "hydro", "qa", "flood")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out_root / "pipeline.log")])

    steps = args.steps or [s for s in ALL_STEPS if cfg.get(s, {}).get("enabled", False)]
    log.info("Project '%s' -> %s | steps: %s", cfg["project_name"], out_root, steps)
    t0 = time.time()

    # --- AOI and analysis grid (always rebuilt; cheap and deterministic) ---
    aoi_4326 = load_aoi(cfg["aoi"])
    grid, aoi_utm, buffered_utm, dl_bounds = build_grid(
        aoi_4326, cfg["aoi"].get("buffer_km", 0), cfg["grid"]["resolution"], cfg["grid"].get("crs", "auto"))
    save_geojson(aoi_4326, WGS84, out_root / "aoi.geojson")
    save_geojson(buffered_utm, grid.crs, out_root / "aoi_buffered.geojson")
    grid_info = {"crs": grid.crs_code, "x0": grid.x0, "y0": grid.y0, "res": grid.res,
                 "width": grid.width, "height": grid.height, "download_bounds_4326": dl_bounds}
    (out_root / "grid.json").write_text(json.dumps(grid_info, indent=2))
    area_km2 = grid.width * grid.height * grid.res ** 2 / 1e6
    log.info("Grid %s %dx%d @ %gm (%.1f km2 incl. buffer)", grid.crs_code,
             grid.width, grid.height, grid.res, area_km2)

    report = {"config": cfg, "grid": grid_info, "outputs": {}}

    # --- Source 1: AWS ---
    if "aws" in steps:
        from sources import fetch_glo30_aws, warp_to_grid
        raw = fetch_glo30_aws(dl_bounds, dirs["raw"], cfg["aws"].get("cache_tiles", False))
        utm = warp_to_grid(raw, grid, dirs["dem"] / "dem_glo30_aws_utm.tif")
        report["outputs"]["glo30_aws"] = {"raw_4326": str(raw), "utm": str(utm)}

    # --- Source 2: GEE ---
    if "gee" in steps:
        from sources import export_gee_drive, fetch_gee_direct, init_ee
        g = cfg["gee"]
        ee = init_ee(g["project"])
        for key in g["datasets"]:
            if g.get("mode", "direct") == "drive":
                tid = export_gee_drive(ee, key, grid, dl_bounds, g["drive_folder"], cfg["project_name"])
                report["outputs"][f"{key}_gee"] = {"drive_task": tid}
            else:
                path = fetch_gee_direct(ee, key, grid, dl_bounds, dirs["dem"],
                                        g.get("chunk_px", 2048), g.get("workers", 4))
                report["outputs"][f"{key}_gee"] = {"utm": str(path)}

    # --- Import external DEM GeoTIFF(s), e.g. a GEE Drive export ---
    if args.import_dem:
        from sources import import_dems
        paths = sorted({p for pat in args.import_dem for p in (glob.glob(pat) or [pat])})
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"--import-dem: not found: {missing}")
        dst = import_dems(paths, grid, dirs["dem"] / f"dem_{args.import_name}_utm.tif")
        report["outputs"][args.import_name] = {"sources": paths, "utm": str(dst)}

    # --- QA ---
    dem_files = {p.stem.replace("dem_", "").replace("_utm", ""): p
                 for p in sorted(dirs["dem"].glob("dem_*_utm.tif"))}
    if "qa" in steps and dem_files:
        from qa import run_qa
        qa = run_qa(dem_files)
        (dirs["qa"] / "qa_report.json").write_text(json.dumps(qa, indent=2))
        for c in qa["comparisons"]:
            if "rmse" in c:
                log.info("QA %-28s mean=%+.2f  RMSE=%.2f  max|d|=%.2f m",
                         c["pair"], c["mean_diff"], c["rmse"], c["max_abs"])
        report["qa"] = qa

    # --- Hydrology ---
    if "hydro" in steps:
        from hydro import run_hydro
        key = cfg["hydro"]["input"]
        dem_path = dirs["dem"] / f"dem_{key}_utm.tif"
        if not dem_path.exists():
            raise FileNotFoundError(f"{dem_path} not found. Run the step that produces '{key}' first.")
        products = run_hydro(dem_path, dirs["hydro"], cfg["hydro"], grid, aoi_utm)
        report["outputs"]["hydro"] = products
        report["hydro_summary"] = {k: raster_summary(products[k])
                                   for k in ("dem_cond", "hand", "twi", "sink_depth")}

    # --- Flood post-processing of the hydro GeoTIFFs ---
    if "flood" in steps:
        from flood import run_flood
        if not (dirs["hydro"] / "hand.tif").exists():
            raise FileNotFoundError("hydro/hand.tif not found. Run the hydro step first.")
        products, summary = run_flood(dirs["hydro"], dirs["flood"], cfg["flood"], aoi_utm)
        report["outputs"]["flood"] = products
        report["flood_summary"] = summary

    report["elapsed_s"] = round(time.time() - t0, 1)
    (out_root / "run_report.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("Done in %.1fs -> %s", report["elapsed_s"], out_root / "run_report.json")


if __name__ == "__main__":
    main()
