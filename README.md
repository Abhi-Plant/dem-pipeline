# DEM pipeline: Copernicus GLO-30 for flood and drainage analysis

The pipeline downloads Copernicus GLO-30 for any AOI from **AWS Open Data**
(no account needed) and/or **Google Earth Engine** (GLO-30, FABDEM, NASADEM).
It puts every DEM on one common UTM 30 m grid, then builds hydrologically
conditioned flood and drainage layers with WhiteboxTools.

```
config.yaml ──► aoi.py ──► AOI + buffer ──► snapped UTM grid (shared by every source)
                              │
          ┌───────────────────┴───────────────────┐
   sources.py: AWS COGs                   sources.py: GEE computePixels
   (streams only the AOI window)          (chunked, parallel, same grid)
          │                                       │
   raw/dem_glo30_aws_4326.tif             dem/dem_<ds>_gee_utm.tif
   dem/dem_glo30_aws_utm.tif                      │
          └──────────────► qa.py (AWS vs GEE vs FABDEM stats)
                             │
                         hydro.py (WhiteboxTools)
   pit fill → least-cost breach → D8 → flow acc → streams/order → HAND, TWI,
   sink depth, slope, hillshade, optional watersheds
                             │
                         flood.py (post-processing of the GeoTIFFs)
   HAND inundation scenarios · susceptibility classes · ponding zones ·
   area statistics (CSV) · polygons (GPKG) · quick-look map (PNG)
```

## Setup
```bash
pip install -r requirements.txt
earthengine authenticate          # only if you use the GEE step
```
For GEE, set `gee.enabled: true` and `gee.project` to a Cloud project registered for Earth Engine.

## Run
```bash
python run_pipeline.py                                # every step enabled in config.yaml
python run_pipeline.py --steps aws hydro              # AWS only, no GEE
python run_pipeline.py --bbox 77.55 12.90 77.65 13.00 --name blr_east
python run_pipeline.py --steps hydro flood            # rerun hydro with new thresholds
python run_pipeline.py --steps flood                  # only redo flood maps (new levels/classes)
```

### Processing a GeoTIFF from outside the pipeline
Use `--import-dem` for a GEE *Export to Drive* result, a DEM you downloaded yourself, or local LiDAR.
Drive splits big exports into `...-0000000000-0000000000.tif` tiles, and a glob pattern picks them all up.
Tiles are mosaicked, warped onto the analysis grid and saved as `dem/dem_<import-name>_utm.tif`:
```bash
python run_pipeline.py --import-dem "C:/Downloads/dem_fabdem_gee_utm*.tif"        --import-name fabdem_drive --hydro-input fabdem_drive --steps qa hydro flood
```
The AOI can be a bbox, a point with a radius, or a vector file (`aoi.type`).

## Outputs (`outputs/<project>/`)
| File | Use |
|---|---|
| `raw/dem_glo30_aws_4326.tif` | Native 1″ mosaic, archive copy |
| `dem/dem_*_utm.tif` | DEMs on the shared 30 m UTM grid |
| `hydro/dem_cond.tif` | Breached and filled DEM, the base for any hydraulic model |
| `hydro/d8_pointer.tif`, `flow_acc_cells.tif`, `sca.tif` | Flow direction, accumulation, specific catchment area |
| `hydro/streams.tif`, `stream_order.tif`, `streams.gpkg` | Drainage network and Strahler order |
| `hydro/hand.tif` | **Height Above Nearest Drainage**: flood susceptibility (low HAND = floods first) |
| `hydro/twi.tif` | Topographic Wetness Index: saturation and waterlogging |
| `hydro/sink_depth.tif` | Depth of closed depressions in the raw DEM: pluvial ponding and storage |
| `hydro/watersheds.*` | Catchments upstream of `hydro.pour_points` |
| `hydro/aoi_clip/*_aoi.tif` | Main layers clipped to the unbuffered AOI |
| `flood/inundation_depth_<h>m.tif` | HAND inundation for water level *h* above the channel (value = water depth) |
| `flood/flood_susceptibility.tif` | HAND classes 5 = very high … 1 = very low (uint8, 0 = no data) |
| `flood/ponding_depth.tif` | Closed depressions deeper than `ponding_min_depth_m` (pluvial ponding) |
| `flood/flood_summary.csv/.json` | Flooded and ponded area (km², % of AOI), mean depth, ponding volume |
| `flood/flood_polygons.gpkg` | Inundation and ponding polygons, one layer each |
| `flood/flood_quicklook.png` | Susceptibility and inundation map over hillshade |
| `qa/qa_report.json`, `run_report.json` | Stats, AWS-vs-GEE differences, full provenance |

## Notes for flood work
- **GLO-30 is a DSM.** It includes canopy and buildings, which raises urban and forested
  terrain by several metres and creates false dams across drainage lines. For
  inundation or HAND mapping, run the GEE step with `fabdem` and set
  `hydro.input: fabdem_gee`. FABDEM is GLO-30 with forests and buildings removed. It is licensed CC BY-NC-SA,
  so it cannot be used commercially.
- **Breaching vs filling:** least-cost breaching cuts through roads and embankments
  (culverts and bridges) instead of flooding everything upstream. Raise
  `breach_dist_cells` if streams still stop at road crossings. For key
  structures, burn known drainage lines or culverts into the DEM before the hydro step.
- **Edge effects:** flow accumulation is truncated where the catchment crosses
  the raster edge. Use an AOI that covers the full upstream catchment (e.g. a
  HydroBASINS polygon) and keep `buffer_km` > 0.
- **Stream threshold:** `stream_threshold_km2` controls drainage density. HAND
  depends strongly on it, so calibrate against mapped streams or imagery
  (0.1–1 km² is a typical range for 30 m data).
- **Vertical datum:** EGM2008 geoid heights. Convert before you mix with
  ellipsoidal (GNSS) or local-datum survey levels.
- **Flat areas:** water bodies are flattened in GLO-30 (see its WBM/EDM layers).
  HAND and TWI are unreliable on large flats and lakes.
- **Large AOIs:** the GEE `direct` mode pulls 16 MB chunks in parallel. For
  basins of tens of thousands of km², use `gee.mode: drive`. For AWS, set
  `cache_tiles: true` so tiles are downloaded once and reused.
- **HAND inundation is static.** It fills every cell whose HAND is below the water level, whether or not
  the water can actually reach it in time or volume. Use it for screening; for design floods use a
  hydraulic model (HEC-RAS 2D, LISFLOOD-FP) with `hydro/dem_cond.tif` as the terrain.
