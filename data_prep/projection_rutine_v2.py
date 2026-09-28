"""
projection_rutine_v2.py -- project MTG LI-2 AFA onto the IR 10.5 um lat/lon grid.

Changes relative to projection_rutine.py (v1), and why:

1. FIXED PHYSICAL SCALE. v1 divided every AFA image by its own full-disk
   maximum (li_map / max * 255), so a stored value meant a different amount
   of lightning in every image. v2 stores the AFA value itself,
   min(AFA, 255), identical in meaning in every file. "Any lightning" is
   AFA >= 1; any threshold is now a fixed number of flash coverages.

2. AGGREGATION, NOT INTERPOLATION. v1 linearly interpolated the 2 km FCI
   grid onto the ~3.14 km lat/lon grid via a Delaunay triangulation of all
   5568 x 5568 points, which samples the field at target points and can
   miss or smear isolated lightning pixels. AFA is sparse (one record per
   lit pixel), so v2 maps each record to the target pixel that contains its
   centre and keeps the MAXIMUM (max, not sum: neighbouring 2 km pixels
   covered by the same flash would otherwise be double counted). Also far
   faster: no triangulation.

3. CORRECT GRID SPACING. v1 built target coordinates with
   np.linspace(ul, ul + W*px, W), whose spacing is px*W/(W-1), not px. At
   the Central-Africa regions (column ~3300 of 5726) that shifted LI by
   ~0.6 px relative to IR. v2 uses the world-file convention exactly:
   x_luc/y_luc are the CENTRE of the upper-left pixel,
   col = rint((lon - x_luc) / x_size), row = rint((lat - y_luc) / y_size).

4. LOSSLESS OUTPUT. v1 saved LI as JPEG (lossy, ringing around sparse
   spikes). v2 saves 8-bit PNG.

5. NO INTEGER TRUNCATION BEFORE SAVING. v1 cast to uint8 after scaling,
   erasing values < 1 (single-flash pixels whenever the disk max > 255).
   AFA is already an integer count, so nothing is lost now except values
   above 255, which are clipped and counted in the log.

Geolocation of the AFA grid is IDENTICAL to v1 (same azimuth/elevation ->
lat/lon routine and the same hard-coded FCI 2 km grid constants), since v1's
geolocation was validated visually against the IR imagery.

Output: <METEOSAT_ROOT>/afa_projected_v2/<date>/<ir name with band->AFA>.png
(full-disk, same size as the IR image; mostly zeros, so PNG stays small).

Usage:
  python projection_rutine_v2.py                  # all dates, skip existing
  python projection_rutine_v2.py --overwrite
  python projection_rutine_v2.py --dates 2025_11_05 2025_11_06
"""
import argparse
import glob
import logging
import os
import re
import zipfile
from datetime import datetime, timedelta

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

METEOSAT_ROOT_FOLDER = "/media/vladlanda/DATA/Meteosat"
LI_FOLDER = "afa"
IR_FOLDER = "ir_105"
OUT_FOLDER = "afa_projected_v2"

DATE_FOLDER_FORMAT = "%Y_%m_%d"
WLD_DATE_FORMAT = "%Y%m%dT%H%M"
NC_DATE_FORMAT = "%Y%m%d%H%M%S"
WLD_REG_EXPRESSION = r"\d{8}T\d{6}Z{1}_\d{8}T\d{6}Z{1}"

# FCI 2 km fixed grid (same constants as v1's generate_full_LI_coverage_matrix)
FCI_LAMBDA_0 = 1.55561889270898e-01
FCI_F_0 = -1.55561889270898e-01
FCI_STEP = 5.58871526031607e-05
FCI_N = 5568


# --------------------------------------------------------------- geolocation
def azel_to_latlon(az, el, r_eq=6378137.0, f=1 / 298.257223563,
                   h=6378137.0 + 35786400.0, lambda_d=0.0):
    """Identical to v1 LInetCDF._azel_to_latlon."""
    r_pol = r_eq * (1 - f)
    s_4 = r_eq ** 2 / r_pol ** 2
    s_5 = h ** 2 - r_eq ** 2
    sd = np.sqrt(np.clip((h * np.cos(az) * np.cos(el)) ** 2
                         - (np.cos(el) ** 2 + s_4 * np.sin(el) ** 2) * s_5, 0, None))
    s_n = (h * np.cos(az) * np.cos(el) - sd) / (np.cos(el) ** 2 + s_4 * np.sin(el) ** 2)
    s_1 = h - s_n * np.cos(az) * np.cos(el)
    s_2 = -s_n * np.sin(az) * np.cos(el)
    s_3 = s_n * np.sin(el)
    s_xy = np.sqrt(s_1 ** 2 + s_2 ** 2)
    lon = np.rad2deg(np.arctan(s_2 / s_1) + lambda_d)
    lat = np.rad2deg(np.arctan(s_4 * s_3 / s_xy))
    return lat, lon


def fci_rowcol_to_latlon(rows, cols):
    """Lat/lon of FCI 2 km grid cells (row=elevation index, col=azimuth index),
    with v1's exact conventions (azimuth sign flip, returned latitude negated).
    Evaluated only for the sparse lit cells -- no full 5568x5568 grid needed."""
    az = -(cols * FCI_STEP - FCI_LAMBDA_0)
    el = rows * FCI_STEP + FCI_F_0
    lat, lon = azel_to_latlon(az, el)
    return -lat, lon


# --------------------------------------------------------------- AFA reading
def read_afa_records(nc_path):
    """Sparse AFA records: (rows, cols, values) on the FCI 2 km grid."""
    from netCDF4 import Dataset
    with Dataset(nc_path) as nc:
        if nc.type != "AFA":
            raise ValueError(f"{nc_path}: expected type AFA, got {nc.type}")
        x, y = nc.variables["x"], nc.variables["y"]
        cols = np.rint((np.asarray(x[:]) - x.add_offset) / x.scale_factor).astype(np.int64)
        rows = np.rint(-(np.asarray(y[:]) - y.add_offset) / y.scale_factor).astype(np.int64)
        values = np.asarray(nc.variables["accumulated_flash_area"][:], dtype=np.int64)
    return rows, cols, values


def read_wld(wld_path):
    with open(wld_path) as f:
        v = [float(line.strip()) for line in f if line.strip()]
    return {"x_size": v[0], "y_size": v[3], "x_luc": v[4], "y_luc": v[5]}


def project_to_grid(rows, cols, values, wld, shape):
    """Max-aggregate sparse FCI records onto the IR lat/lon grid.

    Returns (grid uint8 = min(AFA,255), stats dict)."""
    height, width = shape
    grid = np.zeros(shape, dtype=np.int64)
    if values.size == 0:
        return grid.astype(np.uint8), {"records": 0, "in_grid": 0, "clipped": 0, "max_afa": 0}
    lat, lon = fci_rowcol_to_latlon(rows.astype(float), cols.astype(float))
    ok = np.isfinite(lat) & np.isfinite(lon)
    tc = np.rint((lon - wld["x_luc"]) / wld["x_size"]).astype(np.int64)
    tr = np.rint((lat - wld["y_luc"]) / wld["y_size"]).astype(np.int64)
    ok &= (tc >= 0) & (tc < width) & (tr >= 0) & (tr < height)
    np.maximum.at(grid, (tr[ok], tc[ok]), values[ok])
    stats = {"records": int(values.size), "in_grid": int(ok.sum()),
             "clipped": int((grid > 255).sum()), "max_afa": int(grid.max())}
    return np.clip(grid, 0, 255).astype(np.uint8), stats


# --------------------------------------------------------------- file matching (as v1)
def get_date_folder_pairs(ir_root, li_root, only_dates=None):
    pairs = []
    for folder in sorted(glob.glob(os.path.join(ir_root, "*"))):
        name = os.path.basename(folder)
        if not os.path.isdir(folder) or (only_dates and name not in only_dates):
            continue
        try:
            datetime.strptime(name, DATE_FOLDER_FORMAT)
        except ValueError:
            continue
        li = os.path.join(li_root, name)
        if os.path.isdir(li):
            pairs.append((folder, li))
    return pairs


def get_files_triples(ir_folder, li_folder):
    """(ir jpg, wld, LI zip) triples; the LI file is the one whose END time is
    IR start + 10 min (same matching rule as v1)."""
    reg = re.compile(WLD_REG_EXPRESSION)
    zips = glob.glob(os.path.join(li_folder, "*.zip"))
    triples = []
    for wld in sorted(glob.glob(os.path.join(ir_folder, "*.wld"))):
        jpg = wld[:-4] + ".jpg"
        found = reg.findall(wld)
        if not os.path.isfile(jpg) or not found:
            continue
        date = datetime.strptime(found[0][:13], WLD_DATE_FORMAT)
        key = (date + timedelta(minutes=10)).strftime(NC_DATE_FORMAT) + "_N"
        match = [z for z in zips if key in z]
        if match:
            triples.append((jpg, wld, match[0]))
    return triples


def unzip_nc(zip_path):
    nc_path = zip_path[:-4] + ".nc"
    if not os.path.isfile(nc_path):
        with zipfile.ZipFile(zip_path) as z:
            member = [m for m in z.namelist() if m.endswith(".nc") and "BODY" in m][0]
            with z.open(member) as src, open(nc_path, "wb") as dst:
                dst.write(src.read())
    return nc_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=METEOSAT_ROOT_FOLDER)
    p.add_argument("--dates", nargs="+", default=None, help="Date folders (YYYY_MM_DD); default all")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    ir_root = os.path.join(args.root, IR_FOLDER)
    li_root = os.path.join(args.root, LI_FOLDER)
    out_root = os.path.join(args.root, OUT_FOLDER)
    totals = {"files": 0, "clipped_px": 0, "errors": 0}
    for ir_folder, li_folder in get_date_folder_pairs(ir_root, li_root, args.dates):
        triples = get_files_triples(ir_folder, li_folder)
        log.info(f"{os.path.basename(ir_folder)}: {len(triples)} IR/LI pairs")
        for jpg, wld_path, zip_path in triples:
            out = os.path.join(out_root, os.path.basename(ir_folder),
                               os.path.basename(jpg).replace("band", "AFA")[:-4] + ".png")
            if os.path.isfile(out) and not args.overwrite:
                continue
            try:
                with Image.open(jpg) as im:
                    shape = (im.size[1], im.size[0])
                rows, cols, values = read_afa_records(unzip_nc(zip_path))
                grid, st = project_to_grid(rows, cols, values, read_wld(wld_path), shape)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                Image.fromarray(grid, mode="L").save(out)
                totals["files"] += 1
                totals["clipped_px"] += st["clipped"]
                if st["in_grid"] < st["records"]:
                    log.debug(f"{out}: {st['records'] - st['in_grid']} records outside the IR grid")
            except Exception as e:
                totals["errors"] += 1
                log.error(f"{jpg}: {e}")
    log.info(f"done: {totals}")


if __name__ == "__main__":
    main()
