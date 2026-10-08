"""DEM acquisition: Copernicus GLO-30 from AWS Open Data and from Google Earth Engine.

AWS  : s3://copernicus-dem-30m (public, no credentials). These are Cloud-Optimised
       GeoTIFFs, one per 1x1 degree, named after the tile's lower-left corner.
       Ocean tiles do not exist. The service is HTTP 404 for those.
GEE  : COPERNICUS/DEM/GLO30 plus optional FABDEM / NASADEM. Pixels are pulled with
       ee.data.computePixels in chunks on the exact analysis grid, so nothing
       goes through Google Drive unless mode == "drive".

Vertical datum for GLO-30 and FABDEM: EGM2008 geoid (orthometric heights, metres).
"""
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import reproject

from rio_utils import NODATA, write_raster

log = logging.getLogger(__name__)

AWS_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
GDAL_HTTP_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="2",
    VSI_CACHE="TRUE",
)

GEE_DATASETS = {
    "glo30": {"id": "COPERNICUS/DEM/GLO30", "band": "DEM", "collection": True},
    "fabdem": {"id": "projects/sat-io/open-datasets/FABDEM", "band": "b1", "collection": True},
    "nasadem": {"id": "NASA/NASADEM_HGT/001", "band": "elevation", "collection": False},
}


# --------------------------------------------------------------------------- AWS
def glo30_tile_name(lat, lon):
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def glo30_tiles_for_bounds(bounds):
    """Return the names of the 1-degree tiles that intersect (w, s, e, n). Antimeridian not handled."""
    w, s, e, n = bounds
    return [glo30_tile_name(lat, lon)
            for lat in range(math.floor(s), math.ceil(n))
            for lon in range(math.floor(w), math.ceil(e))]


def _tile_url(name):
    return f"{AWS_BASE}/{name}/{name}.tif"


def _tile_exists(session, name):
    r = session.head(_tile_url(name), timeout=30)
    if r.status_code == 200:
        return True
    if r.status_code in (403, 404):
        return False
    r.raise_for_status()
    return False


def _download_file(session, url, dst):
    if dst.exists():
        return dst
    tmp = dst.with_suffix(".part")
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dst)
    return dst


def fetch_glo30_aws(dl_bounds, raw_dir, cache_tiles=False):
    """Mosaic GLO-30 over dl_bounds at native resolution (EPSG:4326).

    Writes raw_dir/dem_glo30_aws_4326.tif and returns its path.
    """
    names = glo30_tiles_for_bounds(dl_bounds)
    with requests.Session() as s:
        present = [n for n in names if _tile_exists(s, n)]
        missing = sorted(set(names) - set(present))
        if missing:
            log.warning("AWS: %d tile(s) absent (open ocean / no data): %s", len(missing), missing)
        if not present:
            raise RuntimeError("No GLO-30 tiles cover this AOI")

        if cache_tiles:
            tile_dir = raw_dir / "tiles"
            tile_dir.mkdir(parents=True, exist_ok=True)
            log.info("AWS: downloading %d full tile(s) to %s", len(present), tile_dir)
            sources = [str(_download_file(s, _tile_url(n), tile_dir / f"{n}.tif")) for n in present]
        else:
            log.info("AWS: streaming AOI window from %d tile(s)", len(present))
            sources = [f"/vsicurl/{_tile_url(n)}" for n in present]

    with rasterio.Env(**GDAL_HTTP_ENV):
        datasets = [rasterio.open(p) for p in sources]
        try:
            # Above 50 degrees latitude, tiles have wider longitude spacing. Use the
            # finest spacing so that no detail is lost when the mosaic is built.
            res = (min(d.res[0] for d in datasets), min(d.res[1] for d in datasets))
            arr, transform = merge(datasets, bounds=dl_bounds, res=res,
                                   nodata=NODATA, dtype="float32")
            crs = datasets[0].crs
        finally:
            for d in datasets:
                d.close()

    out = write_raster(raw_dir / "dem_glo30_aws_4326.tif", arr[0], transform, crs)
    log.info("AWS: raw mosaic -> %s (%dx%d)", out, arr.shape[2], arr.shape[1])
    return out


def warp_to_grid(src_path, grid, dst_path, resampling=Resampling.bilinear):
    """Resample a raster onto the analysis grid (bilinear by default, which suits elevation)."""
    dst = np.full((grid.height, grid.width), NODATA, dtype="float32")
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            src_nodata=src.nodata,
            dst_nodata=NODATA,
            resampling=resampling,
        )
    return write_raster(dst_path, dst, grid.transform, grid.crs)


def import_dems(paths, grid, dst_path, resampling=Resampling.bilinear):
    """Import DEM GeoTIFF(s) produced outside the pipeline onto the analysis grid.

    Typical inputs: a GEE Export-to-Drive result (Drive splits large exports into
    several "...-0000000000-0000000000.tif" tiles), a DEM downloaded by hand,
    or local LiDAR. Tiles sharing one CRS are mosaicked first, so bilinear
    resampling has neighbours across tile seams. Files in different CRSs are
    warped one by one, and the first file with valid data wins where they overlap.
    """
    out = np.full((grid.height, grid.width), NODATA, dtype="float32")
    crss = set()
    for path in paths:
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError(f"{path} has no CRS")
            crss.add(src.crs.to_string())
    if len(paths) > 1 and len(crss) == 1:
        arr, transform = merge(list(paths), nodata=NODATA, dtype="float32")
        reproject(source=arr[0], destination=out, src_transform=transform, src_crs=crss.pop(),
                  dst_transform=grid.transform, dst_crs=grid.crs,
                  src_nodata=NODATA, dst_nodata=NODATA, resampling=resampling)
        log.info("Import: mosaicked %d tile(s) and warped to the analysis grid", len(paths))
        paths = []
    for path in paths:
        tmp = np.full_like(out, NODATA)
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError(f"{path} has no CRS")
            reproject(source=rasterio.band(src, 1), destination=tmp,
                      dst_transform=grid.transform, dst_crs=grid.crs,
                      src_nodata=src.nodata, dst_nodata=NODATA, resampling=resampling)
        fill = (out == NODATA) & (tmp != NODATA)
        out[fill] = tmp[fill]
        log.info("Import: %s -> %.1f%% of grid filled", Path(path).name, 100 * fill.mean())
    covered = 100 * (out != NODATA).mean()
    if covered < 99:
        log.warning("Import: only %.1f%% of the analysis grid has data. Check the input extent.", covered)
    return write_raster(dst_path, out, grid.transform, grid.crs)


# --------------------------------------------------------------------------- GEE
def init_ee(project):
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:
        log.info("GEE: no stored credentials; starting authentication")
        ee.Authenticate()
        ee.Initialize(project=project)
    return ee


def gee_image(ee, key, bounds4326):
    """Bilinear-resampled float mosaic of a GEE DEM, with gaps set to NODATA."""
    if key not in GEE_DATASETS:
        raise ValueError(f"Unknown GEE dataset '{key}'. Options: {list(GEE_DATASETS)}")
    d = GEE_DATASETS[key]
    region = ee.Geometry.Rectangle(list(bounds4326), proj="EPSG:4326", geodesic=False)
    if d["collection"]:
        col = ee.ImageCollection(d["id"]).filterBounds(region).select([d["band"]])
        native = col.first().projection()
        img = col.map(lambda i: i.resample("bilinear")).mosaic().setDefaultProjection(native)
    else:
        img = ee.Image(d["id"]).select([d["band"]]).resample("bilinear")
    return img.rename("elevation").toFloat().unmask(value=NODATA, sameFootprint=False)


def _compute_chunk(ee, img, grid, r0, c0, h, w, retries=5):
    request = {
        "expression": img,
        "fileFormat": "NUMPY_NDARRAY",
        "bandIds": ["elevation"],
        "grid": {
            "dimensions": {"width": w, "height": h},
            "affineTransform": {
                "scaleX": grid.res, "shearX": 0, "translateX": grid.x0 + c0 * grid.res,
                "shearY": 0, "scaleY": -grid.res, "translateY": grid.y0 - r0 * grid.res,
            },
            "crsCode": grid.crs_code,
        },
    }
    for attempt in range(retries):
        try:
            return r0, c0, ee.data.computePixels(request)["elevation"]
        except Exception as exc:  # quota / transient backend errors
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt * 3
            log.warning("GEE chunk (%d,%d) failed (%s); retrying in %ds", r0, c0, exc, wait)
            time.sleep(wait)


def fetch_gee_direct(ee, key, grid, dl_bounds, dem_dir, chunk_px=2048, workers=4):
    """Download a GEE DEM onto the analysis grid using parallel computePixels chunks."""
    img = gee_image(ee, key, dl_bounds)
    out = np.full((grid.height, grid.width), NODATA, dtype="float32")
    jobs = [(r0, c0, min(chunk_px, grid.height - r0), min(chunk_px, grid.width - c0))
            for r0 in range(0, grid.height, chunk_px)
            for c0 in range(0, grid.width, chunk_px)]
    log.info("GEE[%s]: %d chunk(s) for a %dx%d grid", key, len(jobs), grid.width, grid.height)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_compute_chunk, ee, img, grid, *j) for j in jobs]
        for f in as_completed(futures):
            r0, c0, block = f.result()
            out[r0:r0 + block.shape[0], c0:c0 + block.shape[1]] = block.astype("float32")
    path = write_raster(dem_dir / f"dem_{key}_gee_utm.tif", out, grid.transform, grid.crs)
    log.info("GEE[%s]: -> %s", key, path)
    return path


def export_gee_drive(ee, key, grid, dl_bounds, folder, prefix):
    """Submit an Export.image.toDrive task on the analysis grid (for very large AOIs)."""
    img = gee_image(ee, key, dl_bounds)
    minx, miny, maxx, maxy = grid.bounds
    region = ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj=grid.crs_code, geodesic=False)
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=f"{prefix}_{key}"[:100],
        folder=folder,
        fileNamePrefix=f"dem_{key}_gee_utm",
        region=region,
        crs=grid.crs_code,
        crsTransform=[grid.res, 0, grid.x0, 0, -grid.res, grid.y0],
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True, "noData": NODATA},
    )
    task.start()
    log.info("GEE[%s]: Drive export task %s started (folder '%s'). Put the file in dem/ "
             "once it finishes, then run the hydro step.", key, task.id, folder)
    return task.id
