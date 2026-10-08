"""AOI handling: load the area of interest and derive a snapped metric grid.

Hydrological derivatives (slope, flow accumulation, HAND, TWI) need square
pixels in metres, so every source is resampled onto one common UTM grid.
Because that grid is shared, the AWS and GEE products line up pixel for pixel.
"""
import json
import math
from dataclasses import dataclass

import geopandas as gpd
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from shapely.geometry import Point, box, mapping, shape
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

WGS84 = CRS.from_epsg(4326)
# Pad, in degrees, added to the download extent so that bilinear resampling
# has neighbours at the grid edge (5 GLO-30 pixels of 1 arc-second each).
EDGE_PAD_DEG = 5 / 3600


@dataclass
class Grid:
    crs: CRS
    x0: float  # left edge (m)
    y0: float  # top edge (m)
    res: float
    width: int
    height: int

    @property
    def transform(self):
        return from_origin(self.x0, self.y0, self.res, self.res)

    @property
    def bounds(self):
        return (self.x0, self.y0 - self.height * self.res,
                self.x0 + self.width * self.res, self.y0)

    @property
    def crs_code(self):
        return self.crs.to_string()


def utm_crs(lon, lat):
    zone = min(int((lon + 180) // 6) + 1, 60)
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def reproject_geom(geom, src_crs, dst_crs):
    tf = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shp_transform(tf.transform, geom)


def load_aoi(cfg):
    """Return the AOI as a single shapely geometry in EPSG:4326."""
    kind = cfg["type"]
    if kind == "bbox":
        minx, miny, maxx, maxy = cfg["bbox"]
        if not (minx < maxx and miny < maxy):
            raise ValueError("bbox must be [minLon, minLat, maxLon, maxLat]")
        return box(minx, miny, maxx, maxy)
    if kind == "point":
        lon, lat = cfg["point"]
        local = utm_crs(lon, lat)
        x, y = Transformer.from_crs(WGS84, local, always_xy=True).transform(lon, lat)
        circle = Point(x, y).buffer(float(cfg["radius_km"]) * 1000, resolution=32)
        return reproject_geom(circle, local, WGS84)
    if kind == "file":
        gdf = gpd.read_file(cfg["file"])
        if gdf.crs is None:
            raise ValueError(f"{cfg['file']} has no CRS defined")
        return unary_union(gdf.to_crs(WGS84).geometry.values)
    raise ValueError(f"Unknown aoi.type '{kind}' (use bbox | point | file)")


def build_grid(aoi_4326, buffer_km, res, crs_cfg="auto"):
    """Build the snapped analysis grid around the buffered AOI.

    Returns (grid, aoi_utm, buffered_utm, download_bounds_4326).
    """
    c = aoi_4326.centroid
    crs = utm_crs(c.x, c.y) if str(crs_cfg).lower() == "auto" else CRS.from_user_input(crs_cfg)
    if not crs.is_projected:
        raise ValueError("grid.crs must be a projected (metric) CRS")

    aoi_utm = reproject_geom(aoi_4326, WGS84, crs)
    buffered = aoi_utm.buffer(float(buffer_km) * 1000)
    bminx, bminy, bmaxx, bmaxy = buffered.bounds

    x0 = math.floor(bminx / res) * res
    y0 = math.ceil(bmaxy / res) * res
    width = math.ceil((bmaxx - x0) / res)
    height = math.ceil((y0 - bminy) / res)
    grid = Grid(crs=crs, x0=x0, y0=y0, res=float(res), width=width, height=height)

    w, s, e, n = transform_bounds(crs, WGS84, *grid.bounds, densify_pts=21)
    dl_bounds = (w - EDGE_PAD_DEG, s - EDGE_PAD_DEG, e + EDGE_PAD_DEG, n + EDGE_PAD_DEG)
    return grid, aoi_utm, buffered, dl_bounds


def save_geojson(geom, crs, path):
    """Write a geometry to GeoJSON (always stored in EPSG:4326)."""
    g4326 = reproject_geom(geom, crs, WGS84) if CRS.from_user_input(crs) != WGS84 else geom
    fc = {"type": "FeatureCollection",
          "features": [{"type": "Feature", "properties": {}, "geometry": mapping(g4326)}]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fc))


def load_geojson(path, dst_crs):
    fc = json.loads(path.read_text())
    geom = unary_union([shape(f["geometry"]) for f in fc["features"]])
    return reproject_geom(geom, WGS84, dst_crs)
