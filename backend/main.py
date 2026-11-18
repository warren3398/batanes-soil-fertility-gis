import io, json, tempfile, zipfile, os, functools
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pykrige.ok import OrdinaryKriging
from rasterio.transform import from_origin
from shapely.geometry import mapping
from mapping_outputs import diagnostics, predict_grid, write_outputs
import boto3
from botocore.config import Config
app=FastAPI(title="Soil Fertility GIS API")
app.add_middleware(CORSMiddleware,allow_origins=["https://warren3398.github.io","http://localhost:8000"],allow_methods=["*"],allow_headers=["*"])
app.mount("/assets", StaticFiles(directory=Path(__file__).resolve().parents[1] / "assets"), name="assets")

@app.get("/", include_in_schema=False)
def website():
    return FileResponse(Path(__file__).resolve().parents[1] / "index.html")

@app.get("/records.html", include_in_schema=False)
def sample_records():
    return FileResponse(Path(__file__).resolve().parents[1] / "records.html")

@app.get("/health")
def health(): return {"status":"ok","method":"ordinary_kriging","boundaries":"r2_catalog"}
def loadshape(b,folder):
    Path(folder).mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        members = z.infolist()
        if len(members) > 100 or sum(member.file_size for member in members) > 100_000_000:
            raise ValueError("Shapefile archive is too large")
        for n in z.namelist():
            p=Path(n)
            if p.is_absolute() or ".." in p.parts: raise ValueError("Unsafe ZIP path")
        z.extractall(folder)
    paths=list(Path(folder).rglob("*.shp"))
    if len(paths)!=1: raise ValueError("ZIP must contain exactly one .shp")
    g=gpd.read_file(paths[0])
    if g.crs is None: raise ValueError("Shapefile must include a valid .prj")
    return g

def r2client():
    required=["R2_ACCOUNT_ID","R2_ACCESS_KEY_ID","R2_SECRET_ACCESS_KEY"]
    missing=[k for k in required if not os.environ.get(k)]
    if missing: raise HTTPException(503,"R2 boundary storage not configured: "+", ".join(missing))
    return boto3.client("s3",endpoint_url="https://"+os.environ["R2_ACCOUNT_ID"]+".r2.cloudflarestorage.com",aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],region_name="auto",config=Config(signature_version="s3v4"))
def read_boundary(key, *, ignore_geometry=False):
    bucket = os.environ.get("R2_BUCKET", "slopemap-gis-data")
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "boundary.gpkg")
            r2client().download_file(bucket, key, path)
            return gpd.read_file(path, ignore_geometry=ignore_geometry)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, "Cannot load administrative boundaries from R2; check object key and access") from exc


def field(frame, *aliases):
    columns = {str(column).strip().casefold(): column for column in frame.columns}
    return next((columns[alias] for alias in aliases if alias in columns), None)


@functools.lru_cache(maxsize=1)
def admin_catalog():
    key = os.environ.get("R2_MUNICIPALITY_BOUNDARY_KEY", "boundaries/municipalities.gpkg")
    municipalities = read_boundary(key)
    if municipalities.crs is None:
        raise HTTPException(503, "Boundary layer has no CRS")

    municipality_field = field(
        municipalities, "municipality", "mun_name", "adm3_en", "adm3_name",
        "city_mun", "city_name", "city", "name_3", "name_2",
    )
    if municipality_field is None:
        raise HTTPException(503, "Municipality name field missing: " + ", ".join(municipalities.columns))

    province_aliases = ("province", "prov_name", "prov", "adm2_en", "adm2_name")
    province_field = field(municipalities, *province_aliases)
    if province_field is None:
        # NAME_2 is a province only when the municipality comes from NAME_3.
        aliases = ("name_2", "name_1") if municipality_field.casefold() == "name_3" else ("name_1",)
        province_field = field(municipalities, *aliases)

    if province_field is not None:
        province_names = municipalities[province_field]
    else:
        code_aliases = ("adm2_pcode", "prov_pcode", "province_code", "psgc_prov")
        child_code = field(municipalities, *code_aliases)
        if child_code is None:
            raise HTTPException(503, "Province name or parent code missing: " + ", ".join(municipalities.columns))
        province_key = os.environ.get("R2_PROVINCE_BOUNDARY_KEY", "boundaries/provinces.gpkg")
        provinces = read_boundary(province_key, ignore_geometry=True)
        parent_code = field(provinces, *code_aliases)
        parent_name = field(provinces, *province_aliases, "name_2", "name_1")
        if parent_code is None or parent_name is None:
            raise HTTPException(503, "Province catalog requires a province name and code")
        codes = provinces[parent_code].astype("string").str.strip().str.upper()
        lookup = dict(zip(codes, provinces[parent_name]))
        province_names = municipalities[child_code].astype("string").str.strip().str.upper().map(lookup)

    catalog = municipalities.assign(
        province=province_names.astype("string").str.strip(),
        municipality=municipalities[municipality_field].astype("string").str.strip(),
    )
    if catalog.province.fillna("").eq("").any() or catalog.municipality.fillna("").eq("").any():
        raise HTTPException(503, "Some boundaries have no province or municipality name")
    return catalog

@app.get("/provinces")
def provinces():
    g=admin_catalog()
    return {"provinces":sorted(g.province.dropna().astype(str).unique().tolist())}
@app.get("/municipalities")
def municipalities(province:str):
    g=admin_catalog()
    a=g[g.province.astype(str).str.casefold()==province.casefold()]
    if a.empty: raise HTTPException(404,"Province not found")
    return json.loads(a.to_crs(4326).to_json())

@app.post("/interpolate")
def interpolate(
    points: UploadFile = File(...), results: UploadFile | None = File(None),
    boundary: UploadFile | None = File(None), dem: UploadFile | None = File(None),
    province: str = Form(""), parameter: str = Form("N"), resolution: int = Form(100),
    model: str = Form("spherical"), lags: int = Form(12), lag_distance: float = Form(0),
    dashboard: bool = Form(False), thresholds: str = Form(""), units: str = Form(""),
    manual_variogram: str = Form(""),
):
    from rasterio.errors import RasterioIOError
    from rasterio.features import geometry_mask

    if not 10 <= resolution <= 5000:
        raise HTTPException(400, "Resolution must be 10–5000 m")
    if model not in {"spherical", "exponential", "gaussian", "linear", "power"} or not 3 <= lags <= 30 or not np.isfinite(lag_distance) or lag_distance < 0:
        raise HTTPException(400, "Invalid variogram settings")
    if parameter not in {"N", "P", "K", "pH"}:
        raise HTTPException(400, "Choose N, P, K, or pH")
    try:
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            if boundary:
                area = loadshape(boundary.file.read(), folder / "boundary")
                if area.empty or not area.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
                    raise ValueError("Study boundary must contain polygons")
            else:
                if not province:
                    raise ValueError("Choose a province or upload a study boundary")
                bounds = admin_catalog()
                area = bounds[bounds.province.str.casefold() == province.casefold()].copy()
                if area.empty:
                    raise ValueError("Province not found in the boundary catalog")
            area = area.to_crs(4326)
            poly = area.geometry.make_valid().union_all()
            suffix = Path(points.filename or "").suffix.lower()
            if suffix == ".zip":
                pts = loadshape(points.file.read(), folder / "points").to_crs(4326)
            elif suffix in {".csv", ".xlsx"}:
                table = read_table(points)
                lon = field(table, "longitude", "lon", "lng", "long")
                lat = field(table, "latitude", "lat")
                if lon is None or lat is None:
                    raise ValueError("Sample table needs latitude and longitude columns (WGS 84)")
                x = pd.to_numeric(table[lon], errors="coerce")
                y = pd.to_numeric(table[lat], errors="coerce")
                if not (x.between(-180, 180) & y.between(-90, 90)).all():
                    raise ValueError("Sample table contains invalid latitude or longitude")
                pts = gpd.GeoDataFrame(table, geometry=gpd.points_from_xy(x, y), crs=4326)
            else:
                raise ValueError("Use CSV, XLSX, or a point Shapefile ZIP")
            if not pts.geometry.geom_type.eq("Point").all():
                raise ValueError("Sampling layer must contain points")
            pts = pts[pts.geometry.within(poly)].copy()
            if results:
                lab = read_table(results)
                point_id = field(pts, "sample_id", "sampleid", "id")
                lab_id = field(lab, "sample_id", "sampleid", "id")
                value_col = field(lab, parameter.casefold())
                if point_id is None or lab_id is None or value_col is None:
                    raise ValueError(f"Samples and lab table need sample_id, and the lab table needs {parameter}")
                ids = lab[lab_id].astype("string").str.strip()
                if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
                    raise ValueError("Laboratory sample IDs must be unique and non-empty")
                values = dict(zip(ids, lab[value_col]))
                pts[parameter] = pts[point_id].astype("string").str.strip().map(values)
            else:
                value_col = field(pts, parameter.casefold())
                if value_col is None:
                    raise ValueError(f"Sample data needs a numerical {parameter} field or a separate lab table")
                pts[parameter] = pts[value_col]
            pts[parameter] = pd.to_numeric(pts[parameter], errors="coerce")
            pts = pts[np.isfinite(pts[parameter])].copy()
            if len(pts) < 8:
                raise ValueError("At least 8 numerical samples inside the study area are required")
            if len(pts) > 500:
                raise ValueError("A run supports up to 500 matched samples")
            if pts.geometry.duplicated().any():
                raise ValueError("Combine samples with identical coordinates before running kriging")
            utm = pts.estimate_utm_crs()
            pts = pts.to_crs(utm)
            area = area.to_crs(utm)
            geom = area.geometry.make_valid().union_all()
            minx, miny, maxx, maxy = geom.bounds
            nx = int(np.ceil((maxx-minx)/resolution))
            ny = int(np.ceil((maxy-miny)/resolution))
            if nx * ny > 350000:
                raise ValueError("Grid too large. Increase the resolution in meters")
            transform = from_origin(minx, maxy, resolution, resolution)
            mask = geometry_mask([mapping(geom)], out_shape=(ny, nx), transform=transform, invert=True)
            rows, cols = np.nonzero(mask)
            if not len(rows):
                raise ValueError("No grid cell centers fall inside the boundary. Use a finer resolution")
            x, y, values = pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy(), pts[parameter].to_numpy()
            settings = dict(variogram_model=model, nlags=lags, pseudo_inv=True)
            fitted, validation = diagnostics(x, y, values, settings, lag_distance, dashboard, manual_variogram)
            ok = OrdinaryKriging(x, y, values, **settings, variogram_parameters=fitted)
            z = np.full(mask.shape, -9999, dtype="float32")
            v = z.copy()
            predict_grid(ok, z, v, rows, cols, minx, maxy, resolution)
            summary = dict(province=province or "Custom study area", parameter=parameter, samples=len(pts),
                           resolution_m=resolution, method="ordinary_kriging", model=model, crs=utm.to_string(),
                           units=units.strip()[:32], validation=validation, variogram_fit="manual" if manual_variogram else "automatic")
            output = write_outputs(folder, pts, area, z, v, mask, transform, summary, dashboard,
                                   thresholds, ok, dem, lag_distance, lags)
            return StreamingResponse(output, media_type="application/zip", headers={
                "Content-Disposition": "attachment; filename=soil_fertility_outputs.zip"
            })
    except (ValueError, KeyError, zipfile.BadZipFile, RasterioIOError, np.linalg.LinAlgError) as exc:
        raise HTTPException(400, str(exc)) from exc


def read_table(upload):
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(upload.file, dtype="string")
    if suffix == ".xlsx":
        return pd.read_excel(upload.file, dtype="string")
    raise ValueError("Use CSV or XLSX for laboratory results")
